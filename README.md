### Install requirements

    pip3 install -r requirements.txt

### Run

* Edit next lines in gst_launch.py

      VIDEO_FILENAME = "/path/to/your/video.mp4"
      LABELS_FILE = os.path.join("data/mscoco_label_map.pbtxt")
      WEIGHTS = "/path/to/your/model/frozen_inference_graph.pb"

* Run from command line

      python3 gst_launch.py
