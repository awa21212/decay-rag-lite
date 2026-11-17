## Docker 里 apt 装不上
2026-10-20

在容器里 `apt-get update` 打 archive.ubuntu.com 报 502 Bad Gateway，换清华镜
像（`sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|'`）后一次通过。

同理，`pip install` 报 `SSL: UNEXPECTED_EOF_WHILE_READING` 不是包不存在，是长连接
被掐；加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 即可。