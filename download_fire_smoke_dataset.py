from roboflow import Roboflow

rf = Roboflow(api_key="un339QSWFrxy0KrDtTHN")
project = rf.workspace("custom-thxhn").project("fire-wrpgm")
dataset = project.version(8).download("yolov8") 