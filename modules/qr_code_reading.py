import cv2


class QRCodeReader:
    """
    A class to load an image and decode a QR code using OpenCV.
    This class can be extended for additional image processing or multiple QR code support.
    """

    def __init__(self, image_path: str):
        """
        Initialize the QRCodeReader with the path to the image file.

        :param image_path: Path to the image file containing the QR code.
        """
        self.image_path = image_path
        self.detector = cv2.QRCodeDetector()

    def load_image(self):
        """
        Loads the image from the provided file path.

        :return: The loaded image.
        :raises ValueError: If the image cannot be loaded.
        """
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Unable to open image: {self.image_path}")
        return image

    def decode_qr(self):
        """
        Detects and decodes the QR code from the image.

        :return: A tuple (data, bbox, rectified_image) where:
                 - data: the decoded text (empty string if none found),
                 - bbox: bounding box of the detected QR code,
                 - rectified_image: the binarized QR code image.
        """
        image = self.load_image()
        data, bbox, rectified_image = self.detector.detectAndDecode(image)
        return data, bbox, rectified_image

    def get_qr_data(self):
        """
        Retrieves the decoded QR code data from the image.

        :return: The decoded QR code text.
        :raises ValueError: If no QR code is detected.
        """
        data, bbox, _ = self.decode_qr()
        if data:
            return data
        else:
            raise ValueError("No QR code detected in the image.")


# Example usage:
if __name__ == "__main__":
    # Path to the QR code BMP file.
    image_path = r"D:\Working\BoardDefectChecker\resources\qr_code.bmp"
    reader = QRCodeReader(image_path)
    try:
        qr_content = reader.get_qr_data()
        print("Decoded QR Code:", qr_content)
    except Exception as e:
        print("Error:", e)
