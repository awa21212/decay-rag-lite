【01 docker 里 apt 报 502】

【症状】容器内 `apt-get update` 报 `502 Bad Gateway`，或 pip 报 `SSL: UNEXPECTED_EOF_WHILE_READING`。

【原因】官方源/长连接在当前网络下不可达，不是包不存在。

【修法】换镜像源：
- apt：把 `archive.ubuntu.com` 换成 `mirrors.tuna.tsinghua.edu.cn`
- pip：`-i https://pypi.tuna.tsinghua.edu.cn/simple`
- 需要走宿主代理时：`-e HTTP_PROXY=http://host.docker.internal:7892`

【怎么验证】换源后 `apt-get update` 返回 0，`pip download` 能下到包。

【代价】镜像可能滞后几天；内网环境要换成自己的源。

【别人可能这么问】docker 里装不上包 / apt 更新失败 / pip 报 SSL EOF / 容器没网