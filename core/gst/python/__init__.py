"""
    To use plugins from command line:
        export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD
"""

# TODO come back to it
from pygst_utils import Gst, GObject
# Gst.init(None)

from .gstplugin_py import GstPluginPy


def get_gstreamer_version() -> str:
    """ Returns Gstreamer version

    Returns:
        str: gstreamer version (ex.: 1.14.2)
    """
    major, minor, micro, _ = Gst.version()
    return "{}.{}.{}".format(major, minor, micro)


def register(class_info):
    """ Registers custom Python Plugin statically

        class_info: class_info (Python)
    """

    def init(plugin, plugin_impl, plugin_name):
        """ Registers plugin as GObject

        Args:
            plugin: object (Python Plugin implementation)
            plugin_impl: class_info (Python)
            plugin_name: str (Plugin name, this name will be used for pipeline construct)
        """
        type_to_register = GObject.type_register(plugin_impl)
        return Gst.Element.register(plugin, plugin_name, 0, type_to_register)

    # Parameters explanation
    # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Plugin.html#Gst.Plugin.register_static
    version = get_gstreamer_version()
    gstlicense = 'LGPL'
    origin = ''
    source = class_info.__gstmeta__[1]
    package = class_info.__gstmeta__[0]
    name = class_info.GST_PLUGIN_NAME
    description = class_info.__gstmeta__[2]

    def init_function(plugin): return init(plugin, class_info, name)

    if not Gst.Plugin.register_static(Gst.VERSION_MAJOR, Gst.VERSION_MINOR,
                                      name, description,
                                      init_function, version, gstlicense,
                                      source, package, origin):
        raise ImportError("Plugin {} not registered".format(name))
    return True

# Register plugins statically
register(GstPluginPy)