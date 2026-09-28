"""Serve the Three.js viewer and local video-generation controls."""


def launch(video_path: str | None = None, output_dir: str | None = None,
           model_dir: str = 'models', cache_dir: str = 'outputs/ui',
           server_name: str = '127.0.0.1', server_port: int = 8080):
    """Open a source video or saved result. Batches use all visible GPUs.

    Set server_name=0.0.0.0 to listen on the server's network interfaces.
    """
    if not 1 <= server_port <= 65535:
        raise ValueError('server_port must be between 1 and 65535.')
    import uvicorn
    from fdanyone.space.server import create_app
    app = create_app(cache_dir, model_dir, video_path=video_path, output_dir=output_dir)
    uvicorn.run(app, host=server_name, port=server_port)


if __name__ == '__main__':
    from fire import Fire
    Fire(launch)
