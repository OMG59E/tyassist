
## Debug使用流程


### 1 启动evb板子的rpc服务

- 挂载DEngine到evb板子
- 进入/DEngine/tyhcp目录, 配置环境变量source env_host.sh a55
- 启动rpc, sh syslink_server.sh

### 2 执行示例

- 进入/DEngine/tyhcp目, 配置环境变量source env_client.sh
- 进入/DEngine/tyassist/examples/dbg_case, python3 sample.py