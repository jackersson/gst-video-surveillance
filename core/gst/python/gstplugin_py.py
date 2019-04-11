"""
Usage

    gst-launch-1.0 videotestsrc! gstplugin_py ! videoconvert ! autovideosink

"""

import logging
import timeit
import traceback
import time

from pygst_utils import get_buffer_size, Gst, GObject


class GstPluginPy(Gst.Element):

    # Metadata Explanation:
    # http://lifestyletransfer.com/how-to-create-simple-blurfilter-with-gstreamer-in-python-using-opencv/
    GST_PLUGIN_NAME = 'gstplugin_py'

    __gstmeta__ = ("Name",
                   "Transform",
                   "Description",
                   "Author")

    __gstmetadata__ = __gstmeta__

    _srctemplate = Gst.PadTemplate.new('src', Gst.PadDirection.SRC,
                                       Gst.PadPresence.ALWAYS,
                                       Gst.Caps.from_string("video/x-raw,format={RGB, RGBA}"))

    _sinktemplate = Gst.PadTemplate.new('sink', Gst.PadDirection.SINK,
                                        Gst.PadPresence.ALWAYS,
                                        Gst.Caps.from_string("video/x-raw,format={RGB, RGBA}"))

    __gsttemplates__ = (_srctemplate, _sinktemplate)

    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        "model": (GObject.TYPE_PYOBJECT,
                  "model",
                  "Contains model that implements process(any_data)",
                  GObject.ParamFlags.READWRITE)
    }

    def __init__(self):
        super(GstPluginPy, self).__init__()

        # Explained:
        # http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/

        # Explanation how to init Pads
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/pads.html
        self.sinkpad = Gst.Pad.new_from_template(self._sinktemplate, 'sink')

        # Set chain function
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/chainfn.html
        self.sinkpad.set_chain_function_full(self.chainfunc, None)

        # Set event function
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/eventfn.html
        self.sinkpad.set_event_function_full(self.eventfunc, None)
        self.add_pad(self.sinkpad)

        self.srcpad = Gst.Pad.new_from_template(self._srctemplate, 'src')

        # Set event function
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/eventfn.html
        self.srcpad.set_event_function_full(self.srceventfunc, None)

        # Set query function
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/queryfn.html
        self.srcpad.set_query_function_full(self.srcqueryfunc, None)
        self.add_pad(self.srcpad)

        self.model = None

    def chainfunc(self, pad: Gst.Pad, parent, buffer: Gst.Buffer) -> Gst.FlowReturn:
        """
        :param parent: GstPluginPy
        """
        # Get Buffer Width/Height
        success, (width, height) = get_buffer_size(
            self.srcpad.get_current_caps())

        if not success:
            # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.FlowReturn
            return Gst.FlowReturn.ERROR

        try:
            # Do Buffer processing
            if self.model is not None:
                gst_data = {
                    "buffer": buffer,
                    "width": width,
                    "height": height
                }
                self.model.process(timeout=0.1, **gst_data)

        except Exception as e:
            # logging.error(e)
            # traceback.print_exc()
            # return Gst.FlowReturn.ERROR
            pass

        return self.srcpad.push(buffer)

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == 'model':
            return self.model
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == 'model':
            self.model = value
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def eventfunc(self, pad: Gst.Pad, parent, event: Gst.Event) -> bool:
        """ Forwards event to SRC (DOWNSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadEventFunction
        """
        return self.srcpad.push_event(event)

    def srcqueryfunc(self, pad: Gst.Pad, parent, query: Gst.Query) -> bool:
        """ Forwards query bacj to SINK (UPSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadQueryFunction
        """
        return self.sinkpad.query(query)

    def srceventfunc(self, pad: Gst.Pad, parent, event: Gst.Event) -> bool:
        """ Forwards event back to SINK (UPSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadEventFunction
        """
        return self.sinkpad.push_event(event)


# Required for registering plugin dynamically
# Explained:
# http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstPluginPy)
__gstelementfactory__ = (GstPluginPy.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstPluginPy)
