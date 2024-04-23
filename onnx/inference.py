import cv2
import onnx
import numpy as np
import torch
import onnxruntime as ort


def loadImages(image_path):
    # Load the image
    image = cv2.imread(image_path).mean(-1) / 255
    image = np.pad(
        image, [(0, int(np.ceil(s / 8)) * 8 - s) for s in image.shape[:2]]
    ).astype(np.float32)[None, None, :, :]
    return image


def superpoint(image_path, model_path):
    image = loadImages(image_path)
    model = ort.SessionOptions()
    onnx_session = ort.InferenceSession(model_path, model)
    ort_inputs = {onnx_session.get_inputs()[0].name: image}
    ort_outputs = onnx_session.run(None, ort_inputs)
    return ort_outputs


if __name__ == "__main__":
    image_path = "image/image0.png"
    model_path = "weights/superpoint_v1.onnx"
    keypoints = superpoint(image_path, model_path)
