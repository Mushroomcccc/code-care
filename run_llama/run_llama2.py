import os
import asyncio

# 设置运行端口
os.environ.update({'OLLAMA_HOST': 'http://127.0.0.1:6020'})
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


async def run_process(cmd, verb=True, path=None):
    if path:
        os.chdir(path)
    print(f'>>> starting {path or "."}:', *cmd)
    p = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def pipe(lines):
        async for line in lines:
            print(line.strip().decode('utf-8'))

    if verb:
        await asyncio.gather(
            pipe(p.stdout),
            pipe(p.stderr),
        )
    else:
        await asyncio.gather(pipe(p.stdout))


# ✅ 把异步代码包装在主函数中
async def main():
    await asyncio.gather(
        run_process(['ollama', 'serve']),
    )


# ✅ 使用 asyncio.run() 启动主协程
if __name__ == "__main__":
    asyncio.run(main())
