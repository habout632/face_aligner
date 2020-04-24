import argparse
import os

import cv2
import dlib

from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image


class FaceAligner:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def process_image(self, input_image, output_image, scale=2):

        img = cv2.imread(input_image)

        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        height, width = img.shape[:2]
        s_height, s_width = height // scale, width // scale
        img = cv2.resize(img, (s_width, s_height))

        dets = self.detector(img, 1)

        for i, det in enumerate(dets):
            shape = self.predictor(img, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)

            M = get_rotation_matrix(left_eye, right_eye)
            rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

            cropped = crop_image(rotated, det)

            # cv2.imshow('image', cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # if output_image.endswith('.jpg'):
            #     output_image_path = output_image.replace('.jpg', '_%i.jpg' % i)
            # elif output_image.endswith('.png'):
            #     output_image_path = output_image.replace('.png', '_%i.jpg' % i)
            # else:
            #     output_image_path = output_image + ('_%i.jpg' % i)
            #
            output_image_path = os.path.join(output_image,input_image.split("/")[-1] + ('_%i.jpg' % i))
            cv2.imwrite(output_image_path, cropped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align faces in image')
    parser.add_argument('input', type=str, help='')
    parser.add_argument('output', type=str, help='')
    parser.add_argument('--scale', metavar='S', type=int, default=4, help='an integer for the accumulator')
    parser.add_argument('--predictor', type=str, help='', default="/data/datasets/shape_predictor_68_face_landmarks.dat")
    args = parser.parse_args()

    input_image = args.input
    output_image = args.output
    scale = args.scale
    predictor_path = args.predictor

    face_aligner = FaceAligner(predictor_path)

    for image in os.listdir(input_image):
        print(image)
        face_aligner.process_image(os.path.join("input", image), output_image)

