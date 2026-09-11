# MiniQwen-Omni Web 部署

Web 应用支持文本、麦克风/音频、单张图片、多轮上下文、流式文本、Mimi 语音回复和预设音色。推理请求在一张 PPU 上串行执行，其他用户进入最多 8 个请求的队列，避免并发导致显存峰值或缓存串扰。

界面分为“文字 / 图片”和“语音”两个输入页，二者互不依赖。回复形式显示在输入区上方；“仅文字（更快）”会完全跳过 Talker，“文字 + 语音”会同时生成文字和音频。语音可以单独发送：进入“语音”页 → 开始录音 → 停止并等待波形出现 → 点击“发送这段语音”。浏览器通常会阻止未由用户手势触发的自动播放，看到回复后可手动点击播放器的 ▶。

## 1. 在 tmux 中启动

不要把密码直接写在脚本、命令参数或 Git 仓库中。进入 tmux 后以静默方式输入密码：

```bash
tmux new -s miniqwen-web

cd /path/to/miniqwen-omni
read -rsp 'Web password: ' MINIQWEN_WEB_PASSWORD
echo
export MINIQWEN_WEB_PASSWORD
export MINIQWEN_WEB_USERNAME=miniqwen

bash scripts/serve_web.sh
```

默认使用：

- PPU `0`；
- `out/miniqwen_omni_full/checkpoint`；
- BF16 核心权重；
- `0.0.0.0:7860`；
- 用户名 `miniqwen`，密码来自 `MINIQWEN_WEB_PASSWORD`；
- ASR 展示模型放在 CPU，避免与主模型争用 PPU。

看到 `Running on local URL: http://0.0.0.0:7860` 后，按 `Ctrl-b`、再按 `d` 退出 tmux。关闭本地电脑不影响服务；只要 DSW 实例和 tmux 进程仍在运行，模型就会继续提供服务。

重新查看：

```bash
tmux attach -t miniqwen-web
```

停止服务：在 tmux 中按 `Ctrl-c`，然后执行 `exit`。

## 2. 最快的临时公网体验

在启动前增加：

```bash
export MINIQWEN_WEB_SHARE=1
bash scripts/serve_web.sh
```

终端会输出一个 `https://....gradio.live` 地址。该地址通过 HTTPS 隧道回连当前 DSW 实例，适合短期发给朋友测试，但地址是临时的、可能失效，不适合作为长期服务。

如果仅 DSW 服务器创建隧道时需要代理，可以直接给启动脚本指定代理，不必先 `source proxy_on.sh`：

```bash
export MINIQWEN_WEB_PROXY=http://127.0.0.1:7890
export MINIQWEN_WEB_SHARE=1
bash scripts/serve_web.sh
```

该变量只配置服务端进程的出站代理，不能改变朋友所在网络对 `gradio.live` 的可达性。如果访问者仍必须开代理，代码本身无法让这个境外临时域名变成境内可直连地址；应改用下一节的阿里云 DSW 公网映射，或部署一个境内 HTTPS 反向代理。已经成功下载 `frpc` 后可以先取消 `MINIQWEN_WEB_PROXY` 再试一次，避免不必要的代理流量。

## 3. DSW 固定公网入口

若需要稳定地址，在阿里云控制台为 DSW 实例配置自定义公网服务，将公网端口映射到实例的 TCP `7860`，并在安全组中仅放行需要的来源。阿里云文档说明该方式通常涉及 NAT 网关和 EIP 费用，且 DSW 停止后服务也会停止：

<https://help.aliyun.com/zh/pai/custom-services-access-configurations>

公网访问必须使用 HTTPS。若 DSW 公网入口没有代管 TLS，请在前面配置带证书的 Nginx/Caddy 或其他 HTTPS 网关；不要在纯 HTTP 公网上输入密码。Gradio 内建密码适合小范围体验，不包含 MFA、失败锁定或完整的限流能力。

## 4. 常用配置

所有敏感信息保留在环境变量中，以下变量不会写入 checkpoint：

```bash
# 改用 ModelScope 下载后的 BF16 目录
export MINIQWEN_MODEL_PATH=/path/to/miniqwen-omni-bf16

# 改端口或选择另一张物理 PPU
export MINIQWEN_WEB_PORT=9000
export CUDA_VISIBLE_DEVICES=1

# 语音输入超过 30 秒时提高限制（会增加延迟和显存）
export MINIQWEN_WEB_MAX_AUDIO_SECONDS=60

# 禁用仅用于界面展示的 ASR 转写，模型的 audio-to-audio 输入仍然可用
export MINIQWEN_WEB_DISABLE_ASR=1
```

命令行参数可以追加在启动脚本之后，例如：

```bash
bash scripts/serve_web.sh --max-upload-mb 30
```

## 5. 验证与排障

本机检查端口：

```bash
curl -I http://127.0.0.1:7860
```

未登录时出现重定向或未授权响应是正常的。查看进程和 PPU：

```bash
ps -ef | grep '[w]eb_demo_omni.py'
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi
```

推荐依次验证：纯文本、文本语音回复、语音输入、图片输入、多轮对话。当前实现是“文本逐 token 流式显示，音频 codes 生成完成后再由 Mimi 一次解码”，并非全双工通话。

语音回复的状态应显示“完成”，播放器时长必须大于 0 秒。如果播放器仍显示 0 秒，先按 `Ctrl-c` 停止旧服务并重新执行 `bash scripts/serve_web.sh`；tmux 中已经运行的 Python 进程不会自动加载刚更新的代码。
