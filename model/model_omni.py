import os, math, torch, soundfile as sf, librosa, warnings, numpy as np, onnxruntime as ort, logging, contextlib, io
from dataclasses import dataclass
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import Qwen3Config, Qwen3ForCausalLM, SiglipImageProcessor, SiglipVisionModel, logging as hf_logging
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from .model_talker import TalkerBlock, TalkerConfig, TalkerMoEFeedForward, TalkerRMSNorm, precompute_freqs_cis


class OmniConfig(Qwen3Config):
    model_type = "miniqwen-omni"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "miniqwen-omni"
        self.num_talker_hidden_layers = kwargs.get("num_talker_hidden_layers", 6)
        self.talker_hidden_size = kwargs.get("talker_hidden_size", 768)
        self.use_moe = kwargs.get("use_moe", False)
        self.audio_ids = kwargs.get("audio_ids", [])
        self.audio_special_token = kwargs.get("audio_special_token", "<|audio_pad|>")
        self.audio_start_ids = kwargs.get("audio_start_ids", [])
        self.audio_end_ids = kwargs.get("audio_end_ids", [])
        self.audio_start_special_token = kwargs.get("audio_start_special_token", "<|audio_start|>")
        self.audio_end_special_token = kwargs.get("audio_end_special_token", "<|audio_end|>")
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)
        self.audio_vocab_size = kwargs.get("audio_vocab_size", 2112)
        self.audio_codebook_size = kwargs.get("audio_codebook_size", 2048)
        self.audio_bos_token = kwargs.get("audio_bos_token", 2048)
        self.audio_pad_token = kwargs.get("audio_pad_token", 2049)
        self.audio_stop_token = kwargs.get("audio_stop_token", 2050)
        self.audio_spk_token = kwargs.get("audio_spk_token", 2051)
        self.audio_ref_start_token = kwargs.get("audio_ref_start_token", 2052)
        self.audio_ref_end_token = kwargs.get("audio_ref_end_token", 2053)
        self.use_talker_ref_boundaries = kwargs.get("use_talker_ref_boundaries", False)
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)
        self.think_end_ids = kwargs.get("think_end_ids", [])
        self.image_ids = kwargs.get("image_ids", [])
        self.image_special_token = kwargs.get("image_special_token", "<|image_pad|>")
        self.vision_start_ids = kwargs.get("vision_start_ids", [])
        self.vision_end_ids = kwargs.get("vision_end_ids", [])
        self.vision_start_special_token = kwargs.get("vision_start_special_token", "<|vision_start|>")
        self.vision_end_special_token = kwargs.get("vision_end_special_token", "<|vision_end|>")
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        self.image_grid_size = kwargs.get("image_grid_size", 8)
        self.max_images = kwargs.get("max_images", 4)
        self.use_mrope = kwargs.get("use_mrope", False)
        self.use_modality_boundaries = kwargs.get("use_modality_boundaries", False)
        self.mrope_section = kwargs.get("mrope_section", [24, 20, 20])
        self.accept_hidden_layer = kwargs.get("accept_hidden_layer", 14)
        # Checkpoints written before format v5 have no architecture selector.
        # Keep those loadable through the legacy branch while making fresh
        # Qwen-derived MiniQwen configs use the new same-frame predictor.
        legacy_checkpoint = kwargs.get("architectures") == ["MiniQwenOmni"]
        self.audio_head_type = kwargs.get(
            "audio_head_type",
            "legacy_parallel_delay" if legacy_checkpoint else "main_codec_predictor",
        )
        self.code_predictor_num_layers = kwargs.get("code_predictor_num_layers", 2)
        self.code_predictor_hidden_size = kwargs.get("code_predictor_hidden_size", self.talker_hidden_size)
        self.code_predictor_num_attention_heads = kwargs.get("code_predictor_num_attention_heads", 8)
        self.code_predictor_num_key_value_heads = kwargs.get("code_predictor_num_key_value_heads", 4)
        self.code_predictor_intermediate_size = kwargs.get("code_predictor_intermediate_size", 2432)
        self.residual_codec_loss_weight = kwargs.get("residual_codec_loss_weight", 0.3)
        self.main_codec_temperature = kwargs.get("main_codec_temperature", 0.2)
        self.main_codec_top_k = kwargs.get("main_codec_top_k", 50)
        self.main_codec_top_p = kwargs.get("main_codec_top_p", 0.8)
        self.residual_codec_temperature = kwargs.get("residual_codec_temperature", 1.0)
        self.residual_codec_top_k = kwargs.get("residual_codec_top_k", 50)
        self.residual_codec_top_p = kwargs.get("residual_codec_top_p", 0.8)

    @classmethod
    def from_qwen_pretrained(cls, pretrained_path, **overrides):
        values = Qwen3Config.from_pretrained(pretrained_path).to_dict()
        values.pop("model_type", None)
        values.update(overrides)
        return cls(**values)


@dataclass
class OmniCache:
    thinker: DynamicCache | None = None
    talker: list | None = None
    rope_deltas: torch.Tensor | None = None


class MiniQwenMRoPERotaryEmbedding(Qwen3RotaryEmbedding):
    """Qwen3-Omni interleaved MRoPE over an unchanged Qwen3 backbone."""

    def __init__(self, config, device=None):
        super().__init__(config, device=device)
        self.mrope_section = list(config.mrope_section)
        if sum(self.mrope_section) != self.inv_freq.numel():
            raise ValueError(
                f"mrope_section {self.mrope_section} must sum to head_dim/2={self.inv_freq.numel()}"
            )

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            return super().forward(x, position_ids)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(f"MRoPE position_ids must be (3,B,T), got {tuple(position_ids.shape)}")
        inv_freq = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        ).to(x.device)
        expanded_positions = position_ids[:, :, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ expanded_positions).transpose(2, 3)
            interleaved = freqs[0].clone()
            for dim, offset in enumerate((1, 2), start=1):
                end = self.mrope_section[dim] * 3
                interleaved[..., offset:end:3] = freqs[dim, ..., offset:end:3]
            emb = torch.cat((interleaved, interleaved), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class MMAudioProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


class TalkerHead(nn.Module):
    def __init__(self, in_features, out_features, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = nn.ModuleList([nn.Sequential(nn.Linear(in_features, rank, bias=False), nn.GELU(), nn.Linear(rank, out_features, bias=False)) for _ in range(num_layers)])
    def forward(self, x):
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class TalkerEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = nn.ModuleList([nn.Sequential(nn.Embedding(num_embeddings, rank), nn.GELU(), nn.Linear(rank, embedding_dim, bias=False)) for _ in range(num_layers)])
    def forward(self, x):
        base_out = self.base(x)
        return sum(base_out[:, i, :] + self.adapters[i](x[:, i, :]) for i in range(len(self.adapters))) / self.num_layers


class MimiCodecEmbedding(nn.Module):
    """Independent Mimi codebook embeddings with same-frame mean fusion."""

    def __init__(self, audio_vocab_size, codebook_size, hidden_size, num_codebooks=8):
        super().__init__()
        self.codebook_size = codebook_size
        self.main = nn.Embedding(audio_vocab_size, hidden_size)
        self.residual = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_size) for _ in range(num_codebooks - 1)
        ])

    def codebook(self, index, codes):
        if index == 0:
            return self.main(codes)
        return self.residual[index - 1](codes)

    def forward(self, codes):
        if codes.ndim != 3 or codes.size(1) != 8:
            raise ValueError(f"Mimi codec inputs must be (B,8,T), got {tuple(codes.shape)}")
        main_codes = codes[:, 0]
        special = main_codes >= self.codebook_size
        output = self.main(main_codes)
        normal = ~special
        if normal.any():
            normal_codes = codes.permute(0, 2, 1)[normal]
            if ((normal_codes < 0) | (normal_codes >= self.codebook_size)).any():
                raise ValueError("normal Mimi frames must contain eight codes in [0, 2047]")
            frame_embeddings = [self.main(normal_codes[:, 0])]
            frame_embeddings.extend(
                embedding(normal_codes[:, index + 1])
                for index, embedding in enumerate(self.residual)
            )
            output[normal] = torch.stack(frame_embeddings, dim=0).mean(dim=0)
        return output


class CodePredictor(nn.Module):
    """Causal, within-frame predictor for Mimi codebooks c1..c7."""

    def __init__(self, config, codec_embedding):
        super().__init__()
        hidden_size = config.code_predictor_hidden_size
        if hidden_size != config.talker_hidden_size:
            raise ValueError("v1 Code Predictor hidden size must match Talker hidden size")
        predictor_config = TalkerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=config.code_predictor_num_layers,
            num_attention_heads=config.code_predictor_num_attention_heads,
            num_key_value_heads=config.code_predictor_num_key_value_heads,
            head_dim=hidden_size // config.code_predictor_num_attention_heads,
            intermediate_size=config.code_predictor_intermediate_size,
            max_position_embeddings=8,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rope_scaling=None,
            use_moe=False,
        )
        # The same eight embeddings are used for temporal frame feedback and
        # within-frame teacher forcing. Avoid registering a second state-dict
        # path for the shared module.
        object.__setattr__(self, "codec_embedding", codec_embedding)
        self.layers = nn.ModuleList([
            TalkerBlock(index, predictor_config)
            for index in range(config.code_predictor_num_layers)
        ])
        self.norm = TalkerRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, config.audio_codebook_size, bias=False) for _ in range(7)
        ])
        cos, sin = precompute_freqs_cis(
            dim=predictor_config.head_dim,
            end=8,
            rope_base=config.rope_theta,
            rope_scaling=None,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def _run(self, hidden_states, start_pos=0, past_key_values=None, use_cache=False):
        positions = (
            self.freqs_cos[start_pos:start_pos + hidden_states.size(1)],
            self.freqs_sin[start_pos:start_pos + hidden_states.size(1)],
        )
        presents = []
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                positions,
                past_key_value=past,
                use_cache=use_cache,
                attention_mask=None,
            )
            presents.append(present)
        return self.norm(hidden_states), presents

    def forward(self, talker_hidden, frame_codes):
        if frame_codes.ndim != 2 or frame_codes.size(1) != 8:
            raise ValueError(f"Code Predictor targets must be (N,8), got {tuple(frame_codes.shape)}")
        if ((frame_codes < 0) | (frame_codes >= self.codec_embedding.codebook_size)).any():
            raise ValueError("Code Predictor targets must contain only Mimi codes in [0, 2047]")
        teacher_embeddings = [talker_hidden.unsqueeze(1)]
        teacher_embeddings.extend(
            self.codec_embedding.codebook(index, frame_codes[:, index]).unsqueeze(1)
            for index in range(7)
        )
        hidden_states, _ = self._run(torch.cat(teacher_embeddings, dim=1))
        # h_t is a conditioning prefix. E_i(c_i) predicts c_{i+1}.
        return [self.heads[index](hidden_states[:, index + 1]) for index in range(7)]

    @torch.inference_mode()
    def generate(self, talker_hidden, first_code, sample_fn, temperature=1.0, top_k=50, top_p=0.8):
        if talker_hidden.size(0) != 1 or first_code.numel() != 1:
            raise ValueError("cached Code Predictor generation currently supports batch size 1")
        first_embedding = self.codec_embedding.codebook(0, first_code.reshape(1)).unsqueeze(1)
        hidden_states, cache = self._run(
            torch.cat((talker_hidden.unsqueeze(1), first_embedding), dim=1),
            use_cache=True,
        )
        codes = [int(first_code.item())]
        current = hidden_states[:, -1]
        for index in range(7):
            code = sample_fn(self.heads[index](current)[0], temperature, top_k, top_p)
            codes.append(code)
            if index == 6:
                break
            embedding = self.codec_embedding.codebook(
                index + 1,
                torch.tensor([code], dtype=torch.long, device=talker_hidden.device),
            ).unsqueeze(1)
            hidden_states, cache = self._run(
                embedding,
                start_pos=index + 2,
                past_key_values=cache,
                use_cache=True,
            )
            current = hidden_states[:, -1]
        return codes

class SenseVoiceAudioProcessor:
    def __init__(self, frontend): self.frontend = frontend
    def __call__(self, wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, **kwargs):
        if isinstance(wav, np.ndarray): wav = torch.from_numpy(wav).float()
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        return SimpleNamespace(input_features=fbank, attention_mask=(torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0))


class TalkerModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.talker_config = TalkerConfig(
            hidden_size=config.talker_hidden_size,
            use_moe=config.use_moe,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.layers = nn.ModuleList([TalkerBlock(l, self.talker_config) for l in range(config.num_talker_hidden_layers)])
        self.norm = TalkerRMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        audio_head_type = getattr(config, "audio_head_type", "legacy_parallel_delay")
        if audio_head_type == "main_codec_predictor":
            self.embed_tokens = MimiCodecEmbedding(
                config.audio_vocab_size,
                config.audio_codebook_size,
                config.talker_hidden_size,
            )
            self.lm_head = nn.Linear(config.talker_hidden_size, config.audio_vocab_size, bias=False)
            self.code_predictor = CodePredictor(config, self.embed_tokens)
        elif audio_head_type == "legacy_parallel_delay":
            self.lm_head = TalkerHead(config.talker_hidden_size, config.audio_vocab_size)
            self.embed_tokens = TalkerEmbedding(config.audio_vocab_size, config.talker_hidden_size)
            self.code_predictor = None
        else:
            raise ValueError(f"unsupported audio_head_type: {audio_head_type}")
        self.codec_proj = nn.Sequential(nn.Linear(config.talker_hidden_size, config.talker_hidden_size), nn.GELU(), nn.Linear(config.talker_hidden_size, config.talker_hidden_size), TalkerRMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps))
        self.embed_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.talker_hidden_size), TalkerRMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps))
        self.text_scale, self.audio_scale = nn.Parameter(torch.tensor(3.0)), nn.Parameter(torch.tensor(1.0))
        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.talker_config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


class MiniQwenOmni(Qwen3ForCausalLM):
    config_class = OmniConfig

    def _init_weights(self, module):
        """Initialize Qwen modules plus MiniQwen-specific scalar parameters.

        ``from_pretrained`` creates parameters missing from the Qwen checkpoint
        through Transformers' missing-key initialization path.  Explicit
        ``nn.Parameter`` objects are not covered by Qwen's default initializer,
        so initialize the Talker fusion scales here instead of relying only on
        their constructor values (which may be created on the meta device).
        """
        # Use the explicit parent call so this initializer can also be tested
        # without constructing the full 0.6B model.
        Qwen3ForCausalLM._init_weights(self, module)
        if isinstance(module, TalkerModule):
            module.text_scale.data.fill_(3.0)
            module.audio_scale.data.fill_(1.0)

    def __init__(self, config: OmniConfig = None, audio_encoder_path=None, vision_model_path=None):
        config = config or OmniConfig()
        config.architectures = [self.__class__.__name__]
        super().__init__(config)
        object.__setattr__(self, 'thinker', self.model)
        if config.use_mrope:
            self.model.rotary_emb = MiniQwenMRoPERotaryEmbedding(config)
        self.talker = TalkerModule(config)
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size, target_tokens=config.image_token_len)
        self.audio_pad_token = config.audio_pad_token
        self.audio_bos_token = config.audio_bos_token
        self.audio_stop_token = config.audio_stop_token
        self.audio_spk_token = config.audio_spk_token
        self.audio_ref_start_token = config.audio_ref_start_token
        self.audio_ref_end_token = config.audio_ref_end_token
        audio_encoder, audio_processor = self.load_sensevoice(audio_encoder_path)
        object.__setattr__(self, 'audio_encoder', audio_encoder)
        object.__setattr__(self, 'audio_processor', audio_processor)
        vision_encoder, vision_processor = self.load_vision(vision_model_path)
        object.__setattr__(self, 'vision_encoder', vision_encoder)
        object.__setattr__(self, 'vision_processor', vision_processor)
        self.talker.apply(self._init_weights)
        self.audio_proj.apply(self._init_weights)
        self.vision_proj.apply(self._init_weights)

    @staticmethod
    def load_sensevoice(path):
        if path is None or not os.path.exists(path):
            warnings.warn(f"[MiniQwenOmni] SenseVoice path not found: {path}")
            return None, None
        logging.getLogger().setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()):
            from funasr import AutoModel
            m = AutoModel(model=path, trust_remote_code=True, disable_update=True, device="cpu")
        encoder, frontend = m.model.encoder, m.kwargs["frontend"]
        for p in encoder.parameters(): p.requires_grad = False
        return encoder.eval().float(), SenseVoiceAudioProcessor(frontend.eval())

    @torch.compiler.disable
    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        if (audio_inputs is None) or (self.audio_encoder is None) or (not audio_inputs.any()): return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)
        if audio_lens is not None:
            valid_lens = audio_lens[batch_mask].to(valid_fbank.device)
        else:
            valid_lens = torch.tensor([valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device)
        with torch.no_grad():
            emb, _ = self.audio_encoder(valid_fbank, valid_lens)
        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = [self.audio_proj(emb[i, :max(1, min(valid_lens[i].item(), emb.size(1)))].unsqueeze(0).to(proj_dtype)).squeeze(0) for i in range(emb.size(0))]
        if batch_mask.all(): return emb_list
        out = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    @torch.compiler.disable
    def inject_audio_features(self, tokens, h, audio_feats, seqlen):
        if audio_feats is None or not self.config.audio_ids:
            return h
        marker = self.config.audio_ids[0]
        out = []
        for b in range(h.size(0)):
            hb, seq, i = h[b], tokens[b].tolist(), 0
            af = audio_feats[b] if audio_feats[b] is not None else None
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        inject_len = i - start
                        if af.size(0) != inject_len:
                            raise ValueError(f"audio marker/features mismatch: {inject_len} markers vs {af.size(0)} frames")
                        hb = torch.cat((hb[:start], af, hb[start + inject_len:]), dim=0)
                        af = None
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)
    
    @staticmethod
    def load_vision(path):
        if path is None or not os.path.exists(path):
            warnings.warn(f"[MiniQwenOmni] Vision model path not found: {path}. vision_encoder will be None!")
            return None, None
        hf_logging.set_verbosity_error()
        try:
            model = SiglipVisionModel.from_pretrained(path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(path)
        for p in model.parameters():
            p.requires_grad = False
        return model.eval(), processor

    @torch.compiler.disable
    def get_image_embeddings(self, image_inputs):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
            pv = image_inputs['pixel_values']
            if not pv.any():
                return pv.new_zeros(pv.size(0), self.config.image_token_len, self.config.image_hidden_size)
            pixel_attention_mask = image_inputs.get('pixel_attention_mask')
            if pixel_attention_mask is not None and not pixel_attention_mask.any():
                return pv.new_zeros(pv.size(0), self.config.image_token_len, self.config.image_hidden_size)
        with torch.no_grad():
            outputs = self.vision_encoder(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def encode_image_inputs(self, pixel_values):
        if pixel_values is None or self.vision_encoder is None: return None
        mask = pixel_values.flatten(1).any(1)
        proj_dtype = next(self.vision_proj.parameters()).dtype
        if not mask.any():
            return torch.zeros(
                pixel_values.size(0), self.config.image_token_len, self.config.hidden_size,
                dtype=proj_dtype, device=pixel_values.device,
            )
        with torch.no_grad(): emb = self.vision_encoder(pixel_values=pixel_values[mask]).last_hidden_state
        if emb.dim() == 2: emb = emb.unsqueeze(0)
        # The frozen SigLIP encoder is intentionally kept in FP32, while
        # inference/evaluation may load the trainable MiniQwen modules in
        # BF16. PPU LayerNorm requires input and parameter dtypes to match
        # outside autocast, so align explicitly just like the audio projector.
        emb = self.vision_proj(emb.to(dtype=proj_dtype))
        if mask.all(): return emb
        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        return emb.new_zeros(pixel_values.size(0), *emb.shape[1:]).scatter(0, idx, emb)

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        if vf[b][k].size(0) != i - start:
                            raise ValueError(f"image marker/features mismatch: {i - start} markers vs {vf[b][k].size(0)} tokens")
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def build_mrope_position_ids(self, input_ids, attention_mask=None):
        """Build Qwen3-Omni style T/H/W positions for fixed 8x8 image grids."""
        if input_ids.ndim != 2:
            raise ValueError(f"text input_ids must be (B,T), got {tuple(input_ids.shape)}")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = torch.ones(
            (3, input_ids.size(0), input_ids.size(1)),
            dtype=torch.long,
            device=input_ids.device,
        )
        deltas = []
        image_id = self.config.image_ids[0] if self.config.image_ids else None
        vision_start_id = self.config.vision_start_ids[0] if self.config.vision_start_ids else None
        vision_end_id = self.config.vision_end_ids[0] if self.config.vision_end_ids else None
        grid = int(self.config.image_grid_size)
        expected_image_tokens = grid * grid
        if expected_image_tokens != self.config.image_token_len:
            raise ValueError(
                f"image grid {grid}x{grid} does not match image_token_len={self.config.image_token_len}"
            )

        for batch_idx in range(input_ids.size(0)):
            valid_indices = attention_mask[batch_idx].bool().nonzero(as_tuple=False).flatten()
            tokens = input_ids[batch_idx, valid_indices].tolist()
            chunks = []
            cursor = 0
            i = 0
            while i < len(tokens):
                is_image = (
                    vision_start_id is not None
                    and image_id is not None
                    and vision_end_id is not None
                    and tokens[i] == vision_start_id
                    and i + expected_image_tokens + 1 < len(tokens)
                    and all(token == image_id for token in tokens[i + 1:i + 1 + expected_image_tokens])
                    and tokens[i + 1 + expected_image_tokens] == vision_end_id
                )
                if not is_image:
                    chunks.append(torch.full((3, 1), cursor, dtype=torch.long, device=input_ids.device))
                    cursor += 1
                    i += 1
                    continue

                # The boundary itself is a regular 1D token.
                chunks.append(torch.full((3, 1), cursor, dtype=torch.long, device=input_ids.device))
                cursor += 1
                temporal = torch.full((expected_image_tokens,), cursor, dtype=torch.long, device=input_ids.device)
                height = torch.arange(grid, device=input_ids.device).view(-1, 1).expand(grid, grid).reshape(-1) + cursor
                width = torch.arange(grid, device=input_ids.device).view(1, -1).expand(grid, grid).reshape(-1) + cursor
                chunks.append(torch.stack((temporal, height, width)))
                cursor += grid
                chunks.append(torch.full((3, 1), cursor, dtype=torch.long, device=input_ids.device))
                cursor += 1
                i += expected_image_tokens + 2

            sample_positions = torch.cat(chunks, dim=1) if chunks else position_ids.new_zeros((3, 0))
            position_ids[:, batch_idx, valid_indices] = sample_positions
            deltas.append(cursor - len(tokens))
        return position_ids, torch.tensor(deltas, dtype=torch.long, device=input_ids.device).unsqueeze(1)

    def forward(self, input_ids, attention_mask=None, talker_attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, audio_inputs=None, audio_lens=None,
                pixel_values=None, output_audio_logits=True, text_logits_mask=None,
                position_ids=None, audio_targets=None, **args):
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
            text_ids = input_ids
            audio_ids = (
                torch.full((batch_size, 8, seq_length), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
                if output_audio_logits else None
            )
        else:
            if input_ids.ndim != 3 or input_ids.size(1) != 9:
                raise ValueError(f"input_ids must have shape (B,T) or (B,9,T), got {tuple(input_ids.shape)}")
            batch_size, _, seq_length = input_ids.shape
            text_ids, audio_ids = input_ids[:, 8, :], input_ids[:, :8, :]
        if past_key_values is not None and not isinstance(past_key_values, OmniCache):
            raise TypeError("past_key_values must be an OmniCache")
        cache = past_key_values or OmniCache()
        talker_cache = cache.talker or ([None] * len(self.talker.layers))
        if len(talker_cache) != len(self.talker.layers):
            raise ValueError(f"Talker cache has {len(talker_cache)} layers, expected {len(self.talker.layers)}")
        if talker_cache and talker_cache[0] is not None:
            start_pos = talker_cache[0][0].shape[1]
        elif cache.thinker is not None:
            start_pos = cache.thinker.get_seq_length()
        else:
            start_pos = 0
        rope_deltas = cache.rope_deltas
        if self.config.use_mrope and position_ids is None:
            if start_pos == 0:
                position_ids, rope_deltas = self.build_mrope_position_ids(text_ids, attention_mask)
            else:
                if rope_deltas is None:
                    rope_deltas = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
                physical = torch.arange(
                    start_pos, start_pos + seq_length, dtype=torch.long, device=input_ids.device
                ).view(1, -1)
                logical = physical + rope_deltas.to(input_ids.device)
                position_ids = logical.unsqueeze(0).expand(3, -1, -1)
        if output_audio_logits and self.talker.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.talker.talker_config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.talker.freqs_cos, self.talker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
        # ======= Thinker: native Qwen forward =======
        hidden_states = self.model.embed_tokens(text_ids)
        if audio_inputs is not None and start_pos == 0:
            audio_features = self.encode_audio_inputs(audio_inputs, audio_lens)
            hidden_states = self.inject_audio_features(text_ids, hidden_states, audio_features, seq_length)
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                img_emb = self.get_image_embeddings(pixel_values).to(hidden_states.dtype)
                vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(1)
                bs, num, c, im_h, im_w = pixel_values.shape
                vision_tensors = torch.stack([
                    self.encode_image_inputs(pixel_values[:, i, :, :, :])
                    for i in range(num)
                ], dim=1)
            hidden_states = self.count_vision_proj(tokens=text_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=seq_length)
        thinker_attention_mask = attention_mask
        cache_position = None
        if self.config.use_mrope:
            # The stock Qwen3 model uses ``position_ids`` both for RoPE and
            # causal-mask construction. Its mask helper is intentionally 2D,
            # whereas MRoPE positions are (3,B,T). Build the mask with physical
            # cache positions here, then let Qwen3 consume the three logical
            # T/H/W axes only in the rotary embedding.
            cache_position = torch.arange(
                start_pos, start_pos + seq_length, dtype=torch.long, device=input_ids.device
            )
            mask_kwargs = {
                "config": self.model.config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": cache.thinker,
                "position_ids": cache_position.unsqueeze(0),
            }
            thinker_attention_mask = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.model.has_sliding_layers:
                thinker_attention_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        thinker_out = self.model(
            inputs_embeds=hidden_states,
            attention_mask=thinker_attention_mask,
            position_ids=position_ids,
            past_key_values=cache.thinker,
            use_cache=use_cache,
            cache_position=cache_position,
            output_hidden_states=output_audio_logits,
            return_dict=True,
        )
        h_thinker = thinker_out.last_hidden_state

        if text_logits_mask is not None:
            if text_logits_mask.shape != h_thinker.shape[:2]:
                raise ValueError(
                    f"text_logits_mask must be (B,T) aligned with hidden states, got {tuple(text_logits_mask.shape)}"
                )
            # Cross entropy ignores every -100 target.  Projecting only the
            # supervised rows is mathematically equivalent and avoids the very
            # large Qwen vocabulary projection for prompt/padding positions.
            text_logits = self.lm_head(h_thinker[text_logits_mask.bool()])
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            text_logits = self.lm_head(h_thinker[:, slice_indices, :])
        if not output_audio_logits:
            out = MoeCausalLMOutputWithPast(
                aux_loss=h_thinker.new_zeros(()),
                logits=text_logits,
                past_key_values=(
                    OmniCache(thinker=thinker_out.past_key_values, rope_deltas=rope_deltas)
                    if use_cache else None
                ),
            )
            out.audio_logits = None
            return out

        if not 1 <= self.config.accept_hidden_layer <= len(self.model.layers):
            raise ValueError(f"accept_hidden_layer must be in [1, {len(self.model.layers)}]")
        bridge_states = thinker_out.hidden_states[self.config.accept_hidden_layer]

        # ======= Talker: thinker hidden + audio codes, output audio logits =======
        talker_emb = self.talker.embed_tokens(audio_ids)
        spk_emb = args.get('spk_emb', None)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(spk_mask, self.talker.spk_proj(spk_emb).unsqueeze(1), talker_emb)
        hidden_states = self.talker.embed_proj(bridge_states) * self.talker.text_scale + self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        talker_pos_emb = (self.talker.freqs_cos[start_pos:start_pos + seq_length], self.talker.freqs_sin[start_pos:start_pos + seq_length])
        talker_presents = []
        for layer, past_key_value in zip(self.talker.layers, talker_cache):
            hidden_states, present = layer(
                hidden_states,
                talker_pos_emb,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=talker_attention_mask,
            )
            talker_presents.append(present)
        h_talker = self.talker.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        aux_loss = sum(l.mlp.aux_loss for l in self.talker.layers if isinstance(l.mlp, TalkerMoEFeedForward))
        aux_loss += sum(p.sum() for p in self.audio_proj.parameters()) * 0
        aux_loss += sum(p.sum() for p in self.vision_proj.parameters()) * 0
        aux_loss += sum(p.sum() for p in self.talker.spk_proj.parameters()) * 0

        residual_audio_logits = None
        residual_audio_mask = None
        if self.config.audio_head_type == "main_codec_predictor":
            main_audio_logits = self.talker.lm_head(h_talker[:, slice_indices, :])
            audio_logits = main_audio_logits
            if audio_targets is not None:
                if audio_targets.shape != (batch_size, 8, seq_length):
                    raise ValueError(
                        f"audio_targets must be (B,8,T), got {tuple(audio_targets.shape)}"
                    )
                # Only real, complete Mimi frames enter the residual predictor.
                target_frames = audio_targets.permute(0, 2, 1)
                residual_audio_mask = (
                    (target_frames >= 0) &
                    (target_frames < self.config.audio_codebook_size)
                ).all(dim=-1)
                if residual_audio_mask.any():
                    residual_audio_logits = self.talker.code_predictor(
                        h_talker[residual_audio_mask],
                        target_frames[residual_audio_mask],
                    )
                else:
                    # Preserve DDP parameter participation on a rare batch
                    # containing only EOS/no complete audio frame.
                    aux_loss += sum(p.sum() for p in self.talker.code_predictor.parameters()) * 0
        else:
            aux_loss += sum(p.sum() for p in self.talker.lm_head.adapters.parameters()) * 0
            audio_logits = self.talker.lm_head(h_talker[:, slice_indices, :])
            main_audio_logits = None
        
        next_cache = OmniCache(
            thinker=thinker_out.past_key_values,
            talker=talker_presents,
            rope_deltas=rope_deltas,
        ) if use_cache else None
        out = MoeCausalLMOutputWithPast(aux_loss=aux_loss, logits=text_logits, past_key_values=next_cache)
        out.audio_logits = audio_logits
        out.main_audio_logits = main_audio_logits
        out.residual_audio_logits = residual_audio_logits
        out.residual_audio_mask = residual_audio_mask
        out.talker_hidden_states = h_talker[:, slice_indices, :]
        return out

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=None, max_new_tokens=1024, temperature=None, top_p=None,
                 top_k=None, stream=False, rp=1., use_cache=True, return_audio_codes=False, **args):
        eos_token_id = self.generation_config.eos_token_id if eos_token_id is None else eos_token_id
        temperature = self.generation_config.temperature if temperature is None else temperature
        top_p = self.generation_config.top_p if top_p is None else top_p
        top_k = self.generation_config.top_k if top_k is None else top_k
        if stream:
            return self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, top_k, rp, use_cache, return_audio_codes, **args)
        tokens = list(self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, top_k, rp, use_cache, return_audio_codes, **args))
        return tokens[-1] if tokens else input_ids

    @staticmethod
    def _sample_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
        logits = logits.float().clone() / max(float(temperature), 1e-9)
        if top_k and top_k > 0:
            candidate_logits, candidate_ids = torch.topk(logits, min(int(top_k), logits.numel()))
            if top_p and top_p < 1.0:
                mask = torch.cumsum(F.softmax(candidate_logits, dim=-1), dim=-1) > top_p
                mask[1:], mask[0] = mask[:-1].clone(), False
                candidate_logits[mask] = -float('inf')
            sampled = torch.multinomial(F.softmax(candidate_logits, dim=-1), 1)
            return int(candidate_ids[sampled].item())
        if top_p and top_p < 1.0:
            sorted_logits, sorted_ids = torch.sort(logits, descending=True)
            mask = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            mask[1:], mask[0] = mask[:-1].clone(), False
            logits[sorted_ids[mask]] = -float('inf')
        return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())

    def stream_generate(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, top_k, rp, use_cache, return_audio_codes=False, **args):
        if getattr(self.config, "audio_head_type", "legacy_parallel_delay") == "legacy_parallel_delay":
            yield from MiniQwenOmni._stream_generate_legacy(
                self,
                input_ids, eos_token_id, max_new_tokens, temperature, top_p,
                top_k, rp, use_cache, return_audio_codes, **args,
            )
            return
        yield from MiniQwenOmni._stream_generate_main_codec_predictor(
            self,
            input_ids, eos_token_id, max_new_tokens, temperature, top_p,
            top_k, rp, use_cache, return_audio_codes, **args,
        )

    def _stream_generate_main_codec_predictor(self, input_ids, eos_token_id, max_new_tokens,
                                              temperature, top_p, top_k, rp, use_cache,
                                              return_audio_codes=False, **args):
        start_pos = input_ids.shape[1]
        past_kvs = None
        text_finished = False
        audio_finished = not bool(return_audio_codes)
        audio_started = False
        first_text_drain = True
        open_thinking = bool(args.get('open_thinking', False))
        thinking_end_seen = False
        generated_tokens = []
        args.pop('output_audio_logits', None)
        args.pop('logits_to_keep', None)

        audio_buffer = None
        if return_audio_codes:
            audio_buffer = torch.full(
                (1, 8, start_pos), self.audio_pad_token,
                dtype=torch.long, device=input_ids.device,
            )
            spk_emb = args.get('spk_emb')
            ref_codes = args.get('ref_codes')
            ref_len = ref_codes.shape[2] if ref_codes is not None else 0
            spk_reserve = 1 if spk_emb is not None else 0
            ref_start = start_pos
            if self.config.use_talker_ref_boundaries and ref_codes is not None and start_pos - spk_reserve >= 2:
                code_count = min(ref_len, start_pos - spk_reserve - 2)
                ref_start = start_pos - code_count - 2
                audio_buffer[:, 0, ref_start] = self.audio_ref_start_token
                if code_count:
                    audio_buffer[:, :, ref_start + 1:start_pos - 1] = ref_codes[:, :, -code_count:]
                audio_buffer[:, 0, start_pos - 1] = self.audio_ref_end_token
            elif ref_codes is not None:
                ref_start = max(spk_reserve, start_pos - ref_len)
                if ref_start < start_pos:
                    audio_buffer[:, :, ref_start:start_pos] = ref_codes[:, :, -(start_pos - ref_start):]
            if spk_emb is not None and ref_start > 0:
                audio_buffer[:, 0, ref_start - 1] = self.audio_spk_token

        eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
        while input_ids.shape[1] < start_pos + max_new_tokens:
            if return_audio_codes:
                if past_kvs is None or not use_cache:
                    model_input = torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1)
                else:
                    model_input = torch.cat((audio_buffer[:, :, -1:], input_ids[:, -1:].unsqueeze(1)), dim=1)
            else:
                model_input = input_ids if past_kvs is None or not use_cache else input_ids[:, -1:]

            out = self.forward(
                model_input,
                past_key_values=past_kvs,
                use_cache=use_cache,
                logits_to_keep=1,
                output_audio_logits=bool(return_audio_codes),
                **args,
            )
            past_kvs = out.past_key_values

            text_logits = out.logits[0, -1].clone()
            if rp != 1.0:
                seen = list(set(input_ids[0].tolist()))
                scores = text_logits[seen]
                text_logits[seen] = torch.where(scores > 0, scores / rp, scores * rp)
            text_token = self._sample_from_logits(text_logits, temperature, top_k, top_p)
            if text_finished:
                text_token = args.get('newline_token_id') if first_text_drain else args.get(
                    'pad_token_id', self.config.pad_token_id
                )
                if text_token is None:
                    raise ValueError("newline_token_id is required while draining Talker audio")
                first_text_drain = False

            generated_tokens.append(text_token)
            # Dataset BOS is aligned with the first answer token *after* the
            # complete </think> marker. Therefore seeing the marker makes the
            # next generated text token eligible to carry BOS, not this one.
            audio_start_eligible = not open_thinking or thinking_end_seen
            if open_thinking and self.config.think_end_ids:
                thinking_end_seen |= (
                    generated_tokens[-len(self.config.think_end_ids):] == list(self.config.think_end_ids)
                )

            audio_frame = None
            if return_audio_codes and audio_started and not audio_finished:
                main_logits = out.main_audio_logits[0, -1].clone()
                # Main may emit only a real c0 or the audio-only EOS token.
                masked = torch.full_like(main_logits, -float('inf'))
                masked[:self.config.audio_codebook_size] = main_logits[:self.config.audio_codebook_size]
                masked[self.audio_stop_token] = main_logits[self.audio_stop_token]
                c0 = self._sample_from_logits(
                    masked,
                    self.config.main_codec_temperature,
                    self.config.main_codec_top_k,
                    self.config.main_codec_top_p,
                )
                if c0 == self.audio_stop_token:
                    audio_finished = True
                else:
                    audio_frame = self.talker.code_predictor.generate(
                        out.talker_hidden_states[:, -1],
                        torch.tensor([c0], dtype=torch.long, device=input_ids.device),
                        self._sample_from_logits,
                        temperature=self.config.residual_codec_temperature,
                        top_k=self.config.residual_codec_top_k,
                        top_p=self.config.residual_codec_top_p,
                    )

            if not text_finished and text_token in eos_ids:
                text_finished = True

            input_ids = torch.cat((
                input_ids,
                torch.tensor([[text_token]], dtype=torch.long, device=input_ids.device),
            ), dim=1)
            if return_audio_codes:
                next_audio = torch.full(
                    (1, 8, 1), self.audio_pad_token,
                    dtype=torch.long, device=input_ids.device,
                )
                if audio_frame is not None:
                    next_audio[0, :, 0] = torch.tensor(audio_frame, device=input_ids.device)
                elif not audio_started and (audio_start_eligible or text_finished):
                    # BOS is a Main-stream-only control input. The following
                    # Talker step predicts the first real c0.
                    next_audio[0, 0, 0] = self.audio_bos_token
                    audio_started = True
                audio_buffer = torch.cat((audio_buffer, next_audio), dim=2)

            yield (None if text_finished and first_text_drain is False else input_ids[:, start_pos:]), audio_frame
            if text_finished and audio_finished:
                break

    def _stream_generate_legacy(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, top_k, rp, use_cache, return_audio_codes=False, **args):
        start_pos, past_kvs, text_finished, first_finished = input_ids.shape[1], None, False, True
        produce_audio = bool(return_audio_codes)
        args.pop('output_audio_logits', None)
        args.pop('logits_to_keep', None)
        audio_codes = [[] for _ in range(8)] if produce_audio else None
        audio_stop_pos = [None] * 8 if produce_audio else None
        audio_buffer = None
        if produce_audio:
            audio_buffer = torch.full((1, 8, start_pos), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
            spk_emb = args.get('spk_emb', None)
            ref_codes = args.get('ref_codes', None)
            ref_len = ref_codes.shape[2] if ref_codes is not None else 0
            spk_reserve = 1 if spk_emb is not None else 0
            ref_start = start_pos
            if self.config.use_talker_ref_boundaries and ref_codes is not None and start_pos - spk_reserve >= 2:
                code_count = min(ref_len, start_pos - spk_reserve - 2)
                ref_start = start_pos - code_count - 2
                audio_buffer[:, :, ref_start] = self.audio_ref_start_token
                if code_count:
                    audio_buffer[:, :, ref_start + 1:start_pos - 1] = ref_codes[:, :, -code_count:]
                audio_buffer[:, :, start_pos - 1] = self.audio_ref_end_token
            elif ref_codes is not None:
                ref_start = max(spk_reserve, start_pos - ref_len)
                if ref_start < start_pos:
                    audio_buffer[:, :, ref_start:start_pos] = ref_codes[:, :, -(start_pos - ref_start):]
            if spk_emb is not None and ref_start > 0:
                audio_buffer[:, :, ref_start - 1] = self.audio_spk_token
        think_end_step, generated_tokens = None, ([] if args.get('open_thinking', False) else None)
        while input_ids.shape[1] < start_pos + max_new_tokens:
            if produce_audio:
                if past_kvs is None or not use_cache:
                    model_input = torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1)
                else:
                    model_input = torch.cat((audio_buffer[:, :, -1:], input_ids[:, -1:].unsqueeze(1)), dim=1)
            else:
                model_input = input_ids if past_kvs is None or not use_cache else input_ids[:, -1:]
            out = self.forward(
                model_input,
                past_key_values=past_kvs,
                use_cache=use_cache,
                logits_to_keep=1,
                output_audio_logits=produce_audio,
                **args,
            )
            past_kvs = out.past_key_values

            logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
            if rp != 1.0:
                seen = list(set(input_ids[0].tolist())); score = logits[seen]; logits[seen] = torch.where(score > 0, score / rp, score * rp)
            if top_k and top_k > 0:
                candidate_logits, candidate_ids = torch.topk(logits, min(top_k, logits.numel()))
                if top_p and top_p < 1.0:
                    mask = torch.cumsum(F.softmax(candidate_logits, dim=-1), dim=-1) > top_p
                    mask[1:], mask[0] = mask[:-1].clone(), False
                    candidate_logits[mask] = -float('Inf')
                sampled = torch.multinomial(F.softmax(candidate_logits, dim=-1), 1)
                text_token = candidate_ids[sampled].item()
            else:
                if top_p and top_p < 1.0:
                    sorted_l, sorted_i = torch.sort(logits, descending=True)
                    mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
                    mask[1:], mask[0] = mask[:-1].clone(), False
                    logits[sorted_i[mask]] = -float('Inf')
                text_token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

            if text_finished:
                text_token = args.get('newline_token_id') if first_finished else args.get('pad_token_id', self.config.pad_token_id)
                if text_token is None:
                    raise ValueError("newline_token_id is required while draining Talker audio")
                first_finished = False

            step = input_ids.shape[1] - start_pos  # 已生成token数（0=首次，此时模型处理prompt末尾token）
            audio_step = step - 1  # 延迟1步：输出第1个text时无audio，输出第2个text时layer0开始
            if generated_tokens is not None:
                generated_tokens.append(text_token)
                if not think_end_step and generated_tokens[-len(self.config.think_end_ids):] == list(self.config.think_end_ids): think_end_step = step + 2
                audio_step = (step - think_end_step) if think_end_step else -1
            if produce_audio:
                for i, al in enumerate(out.audio_logits):
                    if audio_step < i:
                        audio_codes[i].append(self.audio_pad_token)
                    else:
                        logits_i = al[0, -1, :].clone() / 0.2
                        for prev_code in audio_codes[i][-3:]: score = logits_i[prev_code]; logits_i[prev_code] = torch.where(score > 0, score / 1.05, score * 1.05)
                        top_val, top_idx = logits_i.topk(50)
                        code = top_idx[torch.multinomial(F.softmax(top_val, dim=-1), 1)].item()
                        audio_codes[i].append(code)
                        if audio_stop_pos[i] is None and code >= 2048: audio_stop_pos[i] = len(audio_codes[i]) - 1

            if produce_audio and text_finished and all(audio_stop_pos[i] is not None for i in range(8)): break

            input_ids = torch.cat((input_ids, torch.tensor([[text_token]], device=input_ids.device)), dim=1)
            if produce_audio:
                audio_buffer = torch.cat((audio_buffer, torch.full((1, 8, 1), self.audio_pad_token, dtype=torch.long, device=input_ids.device)), dim=2)
                for i in range(min(audio_step + 1, 8)): audio_buffer[0, i, -1] = audio_codes[i][-1]

            audio_frame = None
            if produce_audio and audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active_layers = sum(1 for i in range(8) if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i])
                if active_layers >= 8: audio_frame = frame
            if not text_finished:
                yield input_ids[:, start_pos:], audio_frame
                eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
                if text_token in eos_ids:
                    text_finished = True
                    if not produce_audio:
                        break
            else:
                yield None, audio_frame


OmniConfig.register_for_auto_class()
MiniQwenOmni.register_for_auto_class("AutoModelForCausalLM")


# ==== Realtime VAD (与模型本体零耦合，纯工程层) ====
class SileroVAD:
    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.h, self.c = np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        self.h[:], self.c[:] = 0, 0

    def __call__(self, chunk, sr=16000):
        out, self.h, self.c = self.session.run(None, {"input": chunk.reshape(1, -1).astype(np.float32), "h": self.h, "c": self.c, "sr": np.array(sr, dtype="int64")})
        return float(out[0][0])


class RealtimeSession:
    def __init__(self, vad_path, sr=16000, threshold=0.8, min_speech_ms=128, min_silence_ms=800):
        self.vad, self.sr, self.threshold = SileroVAD(vad_path), sr, threshold
        self.min_speech, self.min_silence = int(sr * min_speech_ms / 1000), int(sr * min_silence_ms / 1000)
        self.reset()

    def reset(self):
        self.vad.reset()
        self.buffer, self.ring, self.speaking, self.generating, self.interrupt = [], [], False, False, False
        self.speech_samples = self.silence_samples = self.tail_silence = 0

    def push_chunk(self, chunk, W=1024):
        for i in range(0, max(len(chunk), 1), W):
            w = chunk[i:i + W]
            if len(w) < W:
                w = np.pad(w, (0, W - len(w)))
            prob = self.vad(w, self.sr)
            if prob > self.threshold:
                self.silence_samples = self.tail_silence = 0
                self.speech_samples += len(w)
                self.buffer.append(w)
                if self.speech_samples >= self.min_speech and not self.speaking:
                    self.speaking = True
                    self.buffer = self.ring + self.buffer
                    self.ring = []
                if self.generating and self.speaking:
                    self.interrupt = True
                    return 'interrupt'
            elif self.speaking:
                self.silence_samples += len(w)
                self.tail_silence += 1
                self.buffer.append(w)
                if self.silence_samples >= self.min_silence:
                    if self.tail_silence > 1:
                        del self.buffer[-(self.tail_silence - 1):]
                    self.speaking, self.speech_samples, self.silence_samples, self.tail_silence = False, 0, 0, 0
                    return 'speech_end'
            else:
                if self.speech_samples > 0:
                    self.buffer.clear()
                self.speech_samples = 0
                self.ring = [w]
        return 'listening'

    def get_audio(self):
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        return audio
