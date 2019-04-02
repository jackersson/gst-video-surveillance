import argparse
import importlib.util
from types import ModuleType

from pathlib import Path

from core.gst import GstPipelinesController as PipelinesController
from core.gst import GstPipeline as Pipeline


_CONFIG_FILES = {
    str(fl)[len('configs/'):-len('.py')].replace('/', '.'): fl
    for fl in Path('configs').glob('*.py') if not fl.name.startswith('_')
}


def _import_module(module_name: str) -> ModuleType:
    module_path = _CONFIG_FILES[module_name]

    module_spec = importlib.util.spec_from_file_location(
        module_name, module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


def run(config):
    module = _import_module(config)

    video_sources = []
    try:
        video_sources = getattr(module, 'VIDEO_SOURCES')
    except AttributeError:
        return

    pipelines = PipelinesController()

    # Create pipelines
    for i, video_source_config in enumerate(video_sources):
        pipelines.append(Pipeline(video_source_config, index=i))

    # Launch all pipelines
    pipelines.run()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True,
                    help="Path to config without .py")
    args = vars(ap.parse_args())
    run(args['config'])
