from PySide6.QtCore import QThread, Signal
import cv2  # OpenCV for image processing
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt


class ImageProcessor(QThread):
    processing_complete = Signal(str, str)  # (result, status)
    image_display = Signal(np.ndarray)  # Signal to display images

    def __init__(self):
        super().__init__()
        self.image_path = None
        self.is_running = True

        # Directory paths for saving images
        self.raw_images_directory_path = None
        self.resized_images_directory_path = None
        self.binary_images_directory_path = None
        self.edge_images_directory_path = None
        self.ng_images_directory_path = None
        self.ok_images_directory_path = None

        # Config data
        self.config_dict = None

        # Detection results
        self.connector_lock_defect_check = True
        self.jack_fit_defect_check = True

        # Extracted part images
        self.extracted_jack_image_path = None
        self.extracted_jack_image = None
        self.jack_binary_image = None
        self.jack_binary_path = None
        self.pin_detected_image = None
        self.pin_detected_image_path = None

    def set_image_path(self, image_path):
        """Set the path of the image to process."""
        self.image_path = image_path

    def run(self):
        pass

    def preprocessing(self, image_path, show_image=False):
        """Resize the image to 640x480 and save it in the resized images' directory.

        Args:
            image_path (str): Path to the image to be processed.
            show_image (bool): If True, display the processed image.
        """
        print(f"Starting preprocessing for image: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image for preprocessing.")
            return None, None  # Ensure both values are None

        print("Image loaded successfully, resizing...")

        # Resize the image
        resized_image = cv2.resize(image, (640, 480))
        print("Image resized to 640x480.")

        # Construct the path to save the resized image
        if not os.path.exists(self.resized_images_directory_path):
            print(f"Resized images directory does not exist. Creating: {self.resized_images_directory_path}")
            os.makedirs(self.resized_images_directory_path)

        # Create a filename for the resized image
        base_name = os.path.basename(image_path)
        resized_image_path = os.path.join(self.resized_images_directory_path, f"resized_{base_name}")

        # Save the resized image
        success = cv2.imwrite(resized_image_path, resized_image)
        if success:
            print(f"Resized image saved at: {resized_image_path}")
        else:
            print("Failed to save the resized image.")
            return None, None  # Return None for both if saving failed

        # Show the processed image if requested
        if show_image:
            cv2.imshow("Resized Image", resized_image)
            cv2.waitKey(0)  # Allow window to be responsive

        return resized_image_path, resized_image  # Return the path and the resized image

    def convert_to_binary(self, gray_image, part_name, thresholds=[100, 150, 200], show_images=False,
                          apply_erosion=False, apply_dilation=False):
        """
        Convert grayscale image to binary using multiple thresholds, optionally apply erosion and dilation, and save the result.

        Args:
            gray_image (ndarray): Input grayscale image.
            part_name (str): Name of the part (e.g., "connector" or "jack") for naming the output file.
            thresholds (list): List of thresholds to apply.
            show_images (bool): If True, display each binary and processed image.
            apply_erosion (bool): If True, apply erosion to the binary image.
            apply_dilation (bool): If True, apply dilation to the binary image after erosion.

        Returns:
            Tuple[str, ndarray, Optional[ndarray]]: Path to the saved binary image, the binary image, and the processed image (if erosion/dilation is applied).
        """
        binary_images = []
        for threshold in thresholds:
            _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            binary_images.append(binary)
            self.image_display.emit(binary)  # Emit for display each binary image

        # Combine binary images
        combined_binary = np.zeros_like(binary_images[0])
        for binary in binary_images:
            combined_binary = cv2.bitwise_or(combined_binary, binary)

        # Apply erosion if requested
        processed_image = combined_binary
        if apply_erosion:
            kernel = np.ones((3, 3), np.uint8)  # Define the kernel for erosion
            processed_image = cv2.erode(processed_image, kernel, iterations=1)
            if show_images:
                cv2.imshow(f"Eroded Binary Image for {part_name}", processed_image)
                cv2.waitKey(1)  # Allow window to be responsive

        # Apply dilation if requested
        if apply_dilation:
            # kernel = np.ones((3, 3), np.uint8)  # Define the kernel for dilation
            kernel = np.ones((5, 5), np.uint8)  # Define the kernel for dilation
            processed_image = cv2.dilate(processed_image, kernel, iterations=1)
            if show_images:
                cv2.imshow(f"Dilated Binary Image for {part_name}", processed_image)
                cv2.waitKey(1)  # Allow window to be responsive

        # Construct the path to save the combined binary image
        if not os.path.exists(self.binary_images_directory_path):
            print(f"Binary images directory does not exist. Creating: {self.binary_images_directory_path}")
            os.makedirs(self.binary_images_directory_path)

        # Create a timestamped filename for the combined binary image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        binary_image_path = os.path.join(self.binary_images_directory_path, f"{part_name}_binary_image_{timestamp}.png")

        # Save the combined binary image
        success = cv2.imwrite(binary_image_path, combined_binary)
        if success:
            print(f"Binary image saved at: {binary_image_path}")
        else:
            print("Failed to save the binary image.")
            return None, combined_binary, processed_image  # Return None for path if saving failed

        # Show the combined binary image if requested
        if show_images:
            cv2.imshow(f"Combined Binary Image for {part_name}", combined_binary)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed

        return binary_image_path, combined_binary, processed_image  # Return path, binary, and processed images

    def convert_to_binary_single_threshold(self, gray_image, part_name, threshold=150, show_images=False,
                                           apply_erosion=False, apply_dilation=False):
        """
        Convert a grayscale image to a binary image using a single threshold value, optionally apply erosion and dilation, and save the result.

        Args:
            gray_image (ndarray): Input grayscale image.
            part_name (str): Name of the part (e.g., "connector" or "jack") for naming the output file.
            threshold (int): Threshold value to apply.
            show_images (bool): If True, display the binary and processed images.
            apply_erosion (bool): If True, apply erosion to the binary image.
            apply_dilation (bool): If True, apply dilation to the binary image after erosion.

        Returns:
            Tuple[str, ndarray, Optional[ndarray]]: Path to the saved binary image, the binary image, and the processed image (if erosion/dilation is applied).
        """
        # Apply the single threshold to create the binary image
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        self.image_display.emit(binary_image)  # Emit for display the binary image

        # Apply erosion if requested
        processed_image = binary_image
        if apply_erosion:
            kernel = np.ones((3, 3), np.uint8)  # Define the kernel for erosion
            processed_image = cv2.erode(processed_image, kernel, iterations=1)
            if show_images:
                cv2.imshow(f"Eroded Binary Image for {part_name}", processed_image)
                cv2.waitKey(1)  # Allow window to be responsive

        # Apply dilation if requested
        if apply_dilation:
            kernel = np.ones((5, 5), np.uint8)  # Define the kernel for dilation
            processed_image = cv2.dilate(processed_image, kernel, iterations=1)
            if show_images:
                cv2.imshow(f"Dilated Binary Image for {part_name}", processed_image)
                cv2.waitKey(1)  # Allow window to be responsive

        # Construct the path to save the binary image
        if not os.path.exists(self.binary_images_directory_path):
            print(f"Binary images directory does not exist. Creating: {self.binary_images_directory_path}")
            os.makedirs(self.binary_images_directory_path)

        # Create a timestamped filename for the binary image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        binary_image_path = os.path.join(self.binary_images_directory_path, f"{part_name}_binary_image_{timestamp}.png")

        # Save the binary image
        success = cv2.imwrite(binary_image_path, binary_image)
        if success:
            print(f"Binary image saved at: {binary_image_path}")
        else:
            print("Failed to save the binary image.")
            return None, binary_image, processed_image  # Return None for path if saving failed

        # Show the binary image if requested
        if show_images:
            cv2.imshow(f"Binary Image for {part_name}", binary_image)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed

        return binary_image_path, binary_image, processed_image  # Return path, binary, and processed images

    def extract_parts(self, gray_image, show_images=False):
        """
        Extract connector and FPC lead areas from the grayscale image based on configuration coordinates.

        Args:
            gray_image (ndarray): The grayscale image to process.
            show_images (bool): If True, display the extracted images.

        Returns:
            Tuple[str, str]: Paths of the saved images for connector and FPC lead.
        """
        # Retrieve coordinates from config_dict
        connector_coords = (
            self.config_dict["component-1-roi-coordinates"]["top-left-x"],
            self.config_dict["component-1-roi-coordinates"]["top-left-y"],
            self.config_dict["component-1-roi-coordinates"]["bottom-right-x"] -
            self.config_dict["component-1-roi-coordinates"]["top-left-x"],
            self.config_dict["component-1-roi-coordinates"]["bottom-right-y"] -
            self.config_dict["component-1-roi-coordinates"]["top-left-y"]
        )

        fpc_coords = (
            self.config_dict["component-2-roi-coordinates"]["top-left-x"],
            self.config_dict["component-2-roi-coordinates"]["top-left-y"],
            self.config_dict["component-2-roi-coordinates"]["bottom-right-x"] -
            self.config_dict["component-2-roi-coordinates"]["top-left-x"],
            self.config_dict["component-2-roi-coordinates"]["bottom-right-y"] -
            self.config_dict["component-2-roi-coordinates"]["top-left-y"]
        )

        # Extract images using the coordinates
        connector_image = gray_image[connector_coords[1]:connector_coords[1] + connector_coords[3],
                          connector_coords[0]:connector_coords[0] + connector_coords[2]]
        fpc_image = gray_image[fpc_coords[1]:fpc_coords[1] + fpc_coords[3],
                    fpc_coords[0]:fpc_coords[0] + fpc_coords[2]]

        # Create a timestamp for the filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create paths for saving the images with timestamp
        connector_image_path = os.path.join(self.resized_images_directory_path, f"connector_image_{timestamp}.png")
        fpc_image_path = os.path.join(self.resized_images_directory_path, f"fpc_image_{timestamp}.png")

        # Save the extracted images
        cv2.imwrite(connector_image_path, connector_image)
        cv2.imwrite(fpc_image_path, fpc_image)

        print(f"Connector image saved at: {connector_image_path}")
        print(f"FPC image saved at: {fpc_image_path}")

        # Show the extracted images if requested
        if show_images:
            cv2.imshow("Connector Image", connector_image)
            cv2.imshow("FPC Image", fpc_image)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed

        return connector_image_path, fpc_image_path, connector_image, fpc_image  # Return the paths of the saved images

    def add_black_border(self, image, border_size=5):
        """
        Adds a black border around the image.

        Args:
            image (ndarray): The original image.
            border_size (int): The size of the border to add.

        Returns:
            ndarray: The image with a black border added.
        """
        return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                  value=(0, 0, 0))

    def find_contours(self, binary_image, show_image=False):
        """
        Find contours in the binary image and return the contours and information.

        Args:
            binary_image (ndarray): The binary image to process.
            show_image (bool): If True, display the input image with contours drawn.

        Returns:
            List[ndarray]: Contours found in the binary image.
            List[Dict]: Information about each contour (length, area).
        """
        if len(binary_image.shape) == 3:
            gray_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        elif len(binary_image.shape) != 2:
            raise ValueError("Input must be a binary or single-channel grayscale image.")

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_info = []

        for contour in contours:
            length = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            contour_info.append({'length': length, 'area': area})

        if show_image:
            contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            cv2.imshow("Contours", contour_image)
            cv2.waitKey(0)

        return contours, contour_info

    def find_and_draw_longest_contour(self, contours, image, color=(255, 0, 0), thickness=2):
        """
        Finds the bounding box with the largest width and height from the given contours
        and draws it on the image.

        Args:
            contours (List[ndarray]): List of contours to analyze.
            image (ndarray): The image on which to draw the bounding box.
            color (Tuple[int, int, int]): The color of the bounding box in BGR format.
            thickness (int): The thickness of the bounding box lines.

        Returns:
            Tuple[float, Tuple[int, int, int, int], ndarray]:
                The area of the largest bounding box, its bounding box (x, y, w, h), and the modified image.
        """
        largest_box = (0, 0, 0, 0)  # (x, y, w, h)
        max_area = 0  # Initialize maximum area

        for contour in contours:
            if contour.size > 0:  # Ensure the contour is not empty
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h  # Calculate area of the bounding box

                # Check if this bounding box has a larger area than the current max
                if area > max_area:
                    max_area = area
                    largest_box = (x, y, w, h)

        if largest_box != (0, 0, 0, 0):
            # Draw the bounding box on the image
            x, y, w, h = largest_box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            return max_area, largest_box, image
        else:
            print("No contours found.")
            return 0, (0, 0, 0, 0), image  # Return zeroed box coordinates when no contour is found

    def filter_and_draw_bounding_boxes(self, contours, longest_box, image, color=(0, 255, 0), thickness=2):
        """
        Filters bounding boxes that are inside the longest bounding box and draws them on the image.

        Args:
            contours (List[ndarray]): List of contours to analyze.
            longest_box (Tuple[int, int, int, int]): The bounding box (x, y, w, h) of the longest contour.
            image (ndarray): The image on which to draw the bounding boxes.
            color (Tuple[int, int, int]): The color of the bounding boxes in BGR format.
            thickness (int): The thickness of the bounding box lines.

        Returns:
            Tuple[ndarray, List[ndarray]]: The modified image with bounding boxes drawn and a list of filtered contours.
        """
        longest_x, longest_y, longest_w, longest_h = longest_box
        filtered_contours = []

        for contour in contours:
            box = cv2.boundingRect(contour)
            x, y, w, h = box

            # Check if the bounding box is completely inside the longest box
            if (x >= longest_x and y >= longest_y and
                    x + w <= longest_x + longest_w and
                    y + h <= longest_y + longest_h):
                filtered_contours.append(contour)  # Store the original contour

        # Draw the filtered bounding boxes on the image
        for contour in filtered_contours:
            box = cv2.boundingRect(contour)
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

        return image, filtered_contours

    def draw_bounding_boxes(self, image, contours, show_image=False):
        """
        Draw bounding boxes around contours in the image.

        Args:
            image (ndarray): The image on which to draw bounding boxes.
            contours (List[ndarray]): List of contours.
            show_image (bool): If True, display the image with bounding boxes drawn.

        Returns:
            ndarray: The image with bounding boxes drawn.
        """
        boxed_image = image.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if show_image:
            cv2.imshow("Bounding Boxes", boxed_image)
            cv2.waitKey(0)

        return boxed_image

    def find_edges(self, input_image, show_image=False):
        """
        Convert the input image to an edge image using Canny edge detection.

        Args:
            input_image (ndarray): The input image to process.
            show_image (bool): If True, display the processed edge image.

        Returns:
            ndarray: The edge image.
        """
        # Check if the input image is color; if so, convert to grayscale
        if len(input_image.shape) == 3:
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = input_image  # Already grayscale

        # Apply Gaussian Blur to reduce noise and improve edge detection
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, 100, 200)  # Thresholds can be adjusted

        # Show the edge image if requested
        if show_image:
            cv2.imshow("Edge Image", edges)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed

        return edges  # Return the edge image

    def count_pins(self, contours, longest_box_data, height_offset=5, y_offset=5, target_pin_count=5):
        """
        Counts the number of bounding boxes (pins) based on heights within a range and groups
        the remaining boxes based on continuous y-coordinate values.

        Args:
            contours (List[ndarray]): List of contours to analyze.
            longest_box_data (Tuple[int, int, int, int]): The bounding box (x, y, w, h) of the longest contour.
            height_offset (int): Allowed height offset for grouping pin detection.
            y_offset (int): Allowed y-coordinate offset for filtering bounding boxes.
            target_pin_count (int): Desired count of pins to find in a group.

        Returns:
            Tuple[int, List[Tuple[int, int, int, int]]]: The count of detected pins and their bounding boxes.
        """
        longest_x, longest_y, longest_w, longest_h = longest_box_data
        filtered_bounding_boxes = []

        # Print bounding box data before processing
        # print("Bounding boxes data before processing:")
        for contour in contours:
            box = cv2.boundingRect(contour)
            # print(f"Box: {box}")  # Print the (x, y, w, h) of each bounding box
            filtered_bounding_boxes.append(box)

        # Print filtered bounding boxes data
        # print("Filtered bounding boxes data:")
        # for box in filtered_bounding_boxes:
        # print(f"Filtered Box: {box}")  # Print the filtered bounding boxes

        # Remove boxes where width is greater than height
        filtered_bounding_boxes = [
            box for box in filtered_bounding_boxes if box[2] <= box[3]
        ]

        # Print remaining bounding boxes after removal
        # print("Remaining bounding boxes after removing w > h:")
        # for box in filtered_bounding_boxes:
        #    print(f"Remaining Box: {box}")

        # Determine the height range based on the filtered boxes
        if filtered_bounding_boxes:
            heights = [box[3] for box in filtered_bounding_boxes]  # Get heights of remaining boxes
            min_height = min(heights)
            max_height = max(heights)

            # Filter boxes based on height range
            filtered_bounding_boxes = [
                box for box in filtered_bounding_boxes
                if min_height - height_offset <= box[3] <= max_height + height_offset
            ]

            # Print bounding boxes after height filtering
            # print("Bounding boxes after height filtering:")
            # for box in filtered_bounding_boxes:
            #    print(f"Height Filtered Box: {box}")

            # Group boxes based on continuous y values
            filtered_bounding_boxes.sort(key=lambda b: b[1])  # Sort by y-coordinate
            groups = []
            current_group = [filtered_bounding_boxes[0]]

            for box in filtered_bounding_boxes[1:]:
                if abs(box[1] - current_group[-1][1]) <= y_offset:
                    current_group.append(box)
                else:
                    groups.append(current_group)
                    current_group = [box]

            # Add the last group
            if current_group:
                groups.append(current_group)

            # Check for a group that matches the target pin count
            matching_groups = [group for group in groups if len(group) == target_pin_count]

            if matching_groups:
                largest_group = matching_groups[0]  # Take the first matching group
            else:
                # Sort groups by number of elements (size)
                sorted_groups = sorted(groups, key=len, reverse=True)

                # Get the largest group by count and handle the case for groups with more than 2 with at least 5 elements
                if len(sorted_groups) > 1:
                    # Filter groups with at least 5 elements
                    eligible_groups = [group for group in sorted_groups if len(group) >= 5]
                    if len(eligible_groups) > 1:
                        largest_group = eligible_groups[1]  # Get the second largest group
                    else:
                        largest_group = sorted_groups[0]  # If not enough eligible, take the largest
                else:
                    largest_group = sorted_groups[0] if sorted_groups else []

            # Print final groups
            # print("All groups of bounding boxes based on y values:")
            # for group in groups:
            #    print(f"Group: {[box for box in group]}")

        else:
            largest_group = []

        # The count of valid pin bounding boxes
        pin_count = len(largest_group)

        # Print final bounding boxes in the largest group
        # print("Largest group of bounding boxes:")
        # for box in largest_group:
        #    print(f"Largest Box: {box}")

        return pin_count, largest_group

    def show_binary_image_distribution(self, binary_image, part_name):
        """
        Measure and display the distribution of foreground (white) vs. background (black) pixels in a binary image.

        Args:
            binary_image (ndarray): Input binary image (values should be 0 and 255).
            part_name (str): Name of the part associated with the image for display purposes.

        Returns:
            tuple: A tuple containing the percentage of black and white pixels.
        """
        # Calculate the total number of pixels
        total_pixels = binary_image.size

        # Count the number of black (0) and white (255) pixels
        black_pixels = np.count_nonzero(binary_image == 0)
        white_pixels = np.count_nonzero(binary_image == 255)

        # Calculate the percentage
        black_percentage = (black_pixels / total_pixels) * 100
        white_percentage = (white_pixels / total_pixels) * 100

        # Display the distribution
        print(f"Distribution for {part_name}:")
        print(f"  Total Pixels: {total_pixels}")
        print(f"  Black Pixels (Background): {black_pixels} ({black_percentage:.2f}%)")
        print(f"  White Pixels (Foreground): {white_pixels} ({white_percentage:.2f}%)")

        # Visualize as a bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(["Black (Background)", "White (Foreground)"], [black_pixels, white_pixels], color=['blue', 'green'])
        plt.title(f"Pixel Distribution in Binary Image for {part_name}")
        plt.ylabel("Pixel Count")
        # plt.show()

        # Return the percentages as a tuple
        return black_percentage, white_percentage

    def check_connector_lock_defect(self, resized_image, expected_pin_count, expected_top_left_pixel_density,
                                    expected_top_right_pixel_density, left_offset, right_offset):
        """
        Process and evaluate part 1 - Connector - Pin count
        """
        # Extract parts from resized image
        extracted_connector_image_path, self.extracted_jack_image_path, extracted_connector_image, self.extracted_jack_image = self.extract_parts(
            resized_image, show_images=False)

        _extracted_connector_image = extracted_connector_image.copy()

        # Convert extracted connector image to binary with optional dilation
        connector_binary_image_path, connector_binary_image, connector_processed_image = self.convert_to_binary(
            cv2.imread(extracted_connector_image_path, cv2.IMREAD_GRAYSCALE),  # Ensure reading as grayscale
            part_name="connector",  # Specify the part name for the filename
            show_images=False,  # Set to True to display images
            apply_erosion=False,  # Disable erosion
            apply_dilation=True  # Enable dilation
        )

        # Add black border around processed image
        bordered_connector_image = self.add_black_border(connector_processed_image, border_size=5)
        # cv2.imshow("Bordered Connector Image", bordered_connector_image)

        # Find edges from the bordered processed image
        bordered_connector_edges = self.find_edges(bordered_connector_image, show_image=False)

        # Get bordered image contours and their metadata
        bordered_connector_contours, bordered_connector_info = self.find_contours(bordered_connector_edges,
                                                                                  show_image=False)

        # Draw bounding box for the longest contour
        length, longest_box, boxed_connector_image = self.find_and_draw_longest_contour(
            bordered_connector_contours,
            extracted_connector_image,  # or use bordered_connector_image if preferred
            color=(255, 0, 0),  # Blue color in BGR
            thickness=2
        )

        # Optionally display the boxed image
        # cv2.imshow("Longest Contour Boxed", boxed_connector_image)
        # cv2.waitKey(0)

        # print("Length of the longest contour:", length)
        # if longest_box is not None:
        #    print("Bounding box for the longest contour:", longest_box)
        # else:
        #    print("No bounding box drawn.")

        # print("Connector Contours Info:", connector_info)

        # Count pin
        # Find edges from processed image
        connector_edges = self.find_edges(connector_processed_image, show_image=False)

        # Get contours and their metadata
        connector_contours, connector_info = self.find_contours(connector_edges, show_image=False)

        # Draw bounding boxes on dilation image
        # boxed_connector_image = self.draw_bounding_boxes(extracted_connector_image, connector_contours,
        #                                                            show_image=False)

        # Filter and draw bounding boxes inside the longest box
        modified_image, filtered_contours = self.filter_and_draw_bounding_boxes(
            connector_contours,
            longest_box,
            extracted_connector_image,  # Image with the longest contour box already drawn
            color=(0, 255, 0),  # Green color for other bounding boxes
            thickness=2
        )

        # Display the modified image with filtered bounding boxes
        # cv2.imshow("Filtered Bounding Boxes Inside Longest Box", modified_image)
        # cv2.waitKey(0)

        # Count pins based on height and y-coordinate grouping
        _pin_count, _pin_bounding_boxes = self.count_pins(
            connector_contours,
            longest_box,
            height_offset=5,  # Set the height offset as needed
            y_offset=5,  # Set the y-offset as needed
            target_pin_count=12
        )

        print("Number of detected pins:", _pin_count)
        # print("Pin bounding boxes:", _pin_bounding_boxes)

        # Optionally, draw the pin bounding boxes on the image
        for box in _pin_bounding_boxes:
            x, y, w, h = box
            cv2.rectangle(_extracted_connector_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color for pins
            self.pin_detected_image = _extracted_connector_image.copy()

            # Save pin detected image
            # Create a timestamped filename for the image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.pin_detected_image_path = os.path.join(self.binary_images_directory_path,
                                                   f"pin_detected_binary_image_{timestamp}.png")

            # Save the binary image
            success = cv2.imwrite(self.pin_detected_image_path, self.pin_detected_image)
            if success:
                print(f"Pin detected image saved at: {self.pin_detected_image_path}")
            else:
                print("Failed to save the pin detected image.")
        # Display the final image with pins highlighted
        # cv2.imshow("Pins Highlighted", _extracted_connector_image)
        # cv2.waitKey(0)

        if expected_pin_count != _pin_count:
            print("Pin count not match. Detected " + str(_pin_count) + " while expected " + str(expected_pin_count))
            return False

        """
        Process and evaluate part 1 - Connector - Shape correctness
        """
        # From dilated binary image, take longest_box's coordinates as reference, then from this coordinates,
        # crop top left and right corner as squares, size defined by crop_size
        # Count for number of white pixels in crop squares, print to debug
        # Dilated binary image variable name is connector_processed_image

        # Assuming longest_box is defined as (longest_x, longest_y, longest_w, longest_h)
        longest_x, longest_y, longest_w, longest_h = longest_box

        # Define the size of the crop square (can be changed as needed)
        crop_size = 50  # Change this value to adjust the crop size

        # Ensure the crop size does not exceed the dimensions of the image
        if crop_size > longest_w or crop_size > longest_h:
            raise ValueError("Crop size must be less than or equal to the dimensions of the bounding box.")

        # Crop top left corner
        top_left_x = longest_x
        top_left_y = longest_y
        top_left_crop = connector_processed_image[top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]

        # Crop top right corner
        top_right_x = longest_x + longest_w - crop_size
        top_right_y = longest_y
        top_right_crop = connector_processed_image[top_right_y:top_right_y + crop_size,
                         top_right_x:top_right_x + crop_size]

        # Count white pixels in top left crop
        top_left_white_count = cv2.countNonZero(top_left_crop)
        print(f"White pixels in top left corner: {top_left_white_count}")

        # Count white pixels in top right crop
        top_right_white_count = cv2.countNonZero(top_right_crop)
        print(f"White pixels in top right corner: {top_right_white_count}")

        # Show the cropped squares
        # cv2.imshow("Top Left Crop", top_left_crop)
        # cv2.imshow("Top Right Crop", top_right_crop)

        # Wait for a key press and close the image windows
        cv2.waitKey(0)

        if top_left_white_count > expected_top_left_pixel_density + left_offset:
            print("Connector shape correctness check: NG - Left density over allowed")
            return False
        if top_right_white_count > expected_top_right_pixel_density + right_offset:
            print("Connector shape correctness check: NG - Right density over allowed")
            return False

        print("Connector shape correctness check: OK")
        return True

    def check_jack_fit_defect(self):
        """
        Erode to check if FPC is straight - White pixel distribution low
        """
        white_pixel_distribution_percent_threshold = 30

        # Convert to binary image with threshold = 60
        self.jack_binary_path, self.jack_binary_image, jack_processed_image = self.convert_to_binary_single_threshold(
            cv2.imread(self.extracted_jack_image_path),
            part_name="jack",
            threshold=60,
            show_images=False,
            apply_erosion=True,
            apply_dilation=False
        )

        # Check distribution of black-white pixel
        # Define the crop coordinates (x, y, width, height)
        x, y, width, height = 160, 100, 340, 140  # Example coordinates
        # x, y, width, height = 120, 20, 400, 460  # Example coordinates

        # Crop the area from jack_processed_image
        fpc_area_image = jack_processed_image[y:y + height, x:x + width]

        # Optionally, display the cropped image
        # cv2.imshow("FPC Area Image", fpc_area_image)
        # cv2.waitKey(0)

        black_pixel_percent, white_pixel_percent = self.show_binary_image_distribution(fpc_area_image, part_name="jack")

        if white_pixel_percent > white_pixel_distribution_percent_threshold:
            print("White pixel distribution exceeds threshold - NG")
            return False
        else:
            print("White pixel distribution is in allowed range - OK")

        # If pixel percent is very low, we assume that FPC fit defect check is ok. No more check needed
        if white_pixel_percent <= 5:
            print("White pixel distribution is very low - Needn't to check more")
            return True
        """
        Dilate to check if FPC's left and right ears are balanced
        """
        # Convert to binary image with threshold = 60
        _jack_binary_path, _jack_binary_image, _jack_processed_image = self.convert_to_binary_single_threshold(
            cv2.imread(self.extracted_jack_image_path),
            part_name="jack",
            threshold=60,
            show_images=False,
            apply_erosion=False,
            apply_dilation=True
        )

        # Crop the left and right ears
        x_left, y_left, w_left, h_left = 85, 70, 80, 70
        x_right, y_right, w_right, h_right = 455, 70, 80, 70

        # Crop the area which contains both
        x_both, y_both, w_both, h_both = 85, 70, 450, 70
        both_area = _jack_binary_image[y_both:y_both + h_both, x_both:x_both + w_both]

        # Correct the cropping logic
        left_ear = _jack_binary_image[y_left:y_left + h_left, x_left:x_left + w_left]
        right_ear = _jack_binary_image[y_right:y_right + h_right, x_right:x_right + w_right]

        # Optionally, display the cropped image
        # cv2.imshow("Left Ear", left_ear)
        # cv2.waitKey(0)
        # Optionally, display the cropped image
        # cv2.imshow("Right Ear", right_ear)
        # cv2.waitKey(0)
        # Optionally, display the cropped image
        # cv2.imshow("Both Area", both_area)
        # cv2.waitKey(0)

        # Add black border to both area crop image
        bordered_both_area = self.add_black_border(both_area)
        # cv2.imshow("Bordered Both Are", bordered_both_area)
        # cv2.waitKey(0)

        # Find edges from processed image
        jack_edges = self.find_edges(bordered_both_area, show_image=False)

        # Get contours and their metadata
        jack_contours, jack_info = self.find_contours(jack_edges, show_image=False)

        # Draw bounding boxes on dilation image
        boxed_connector_image = self.draw_bounding_boxes(bordered_both_area, jack_contours,
                                                         show_image=False)

        # Extract bounding boxes and append to a list
        jack_bounding_boxes = []
        for contour in jack_contours:
            x, y, w, h = cv2.boundingRect(contour)
            jack_bounding_boxes.append((x, y, w, h))

        # Sort the bounding boxes by height in descending order
        jack_bounding_boxes.sort(key=lambda box: box[3], reverse=True)  # box[3] is height

        # Get the first two highest boxes
        if len(jack_bounding_boxes) >= 2:
            box1 = jack_bounding_boxes[0]
            box2 = jack_bounding_boxes[1]
        else:
            print("Not enough bounding boxes found.")
            return False

        # Compare the y-values of the two boxes
        y_difference = abs(box1[1] - box2[1])  # box[1] is the y-coordinate
        print("Left and right ears difference: " + str(y_difference))
        offset = 5  # Set your desired offset value

        # Print NG or OK based on the y-value comparison
        if y_difference <= offset:
            print("Left and right ears balance check - OK")
            return True
        else:
            print("Left and right ears balance check - NG")
            return False

    def stop(self):
        """Stop the processing thread."""
        self.is_running = False


if __name__ == "__main__":
    image_processor = ImageProcessor()

    # Set config dict
    image_processor.config_dict = {
        "part-name": "Connector - Jack",
        "raw-images-directory-path": r"C:\BoardDefectChecker\images\raw-images",
        "resized-images-directory-path": r"C:\BoardDefectChecker\images\resized-images",
        "binary-images-directory-path": r"C:\BoardDefectChecker\images\binary-images",
        "edge-images-directory-path": r"C:\BoardDefectChecker\images\edge-images",
        "ng-images-directory-path": r"C:\BoardDefectChecker\images\ng-images",
        "ok-images-directory-path": r"C:\BoardDefectChecker\images\ok-images",
        "connection-pin-count": 10,
        "component-1-roi-coordinates": {
            "top-left-x": 0,
            "top-left-y": 0,
            "bottom-right-x": 640,
            "bottom-right-y": 240
        },
        "component-2-roi-coordinates": {
            "top-left-x": 0,
            "top-left-y": 240,
            "bottom-right-x": 640,
            "bottom-right-y": 480
        },
        "pixel-density-edge-1": 150,
        "pixel-density-edge-2": 53,
        "pixel-density-edge-3": 34,
        "tilt-angle-edge-1-2": 60,
        "tilt-angle-edge-2-3": 60
    }

    # Set config directories
    # Directory paths for saving images
    image_processor.raw_images_directory_path = r"C:\BoardDefectChecker\images\raw-images"
    image_processor.resized_images_directory_path = r"C:\BoardDefectChecker\images\resized-images"
    image_processor.binary_images_directory_path = r"C:\BoardDefectChecker\images\binary-images"
    image_processor.edge_images_directory_path = r"C:\BoardDefectChecker\images\edge-images"
    image_processor.ng_images_directory_path = r"C:\BoardDefectChecker\images\ng-images"
    image_processor.ok_images_directory_path = r"C:\BoardDefectChecker\images\ok-images"

    # Set image path to test
    # image_path = r"D:\Working\Images\Grayscale\Sample-3\BoardCheckSample-2.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-3\BoardCheckSample-1.bmp"
    # 180 degree
    # image_path = r"D:\Working\Images\Grayscale\Sample-1\BoardCheckSample-1.bmp"
    # 90 degree
    # image_path = r"D:\Working\Images\Grayscale\Sample-1\BoardCheckSample-2.bmp"
    # 45 degree
    # image_path = r"D:\Working\Images\Grayscale\Sample-1\BoardCheckSample-3.bmp"
    # 30 degree
    # image_path = r"D:\Working\Images\Grayscale\Sample-1\BoardCheckSample-4.bmp"
    # Image from camera on JIG - 180 degree - With light
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-6.bmp"
    # Image from camera on JIG - 180 degree - Without light 1
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-2.bmp"
    # Image from camera on JIG - 180 degree - Without light 2
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-7.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-8.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-9.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-10.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-11.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-12.bmp"
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-13.bmp"
    image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-14.bmp"
    # Image from camera on JIG - 90 degree - Without light
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-3.bmp"
    # Image from camera on JIG - 45 degree - Without light
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-10.bmp"
    # Image from camera on JIG - 30 degree - Without light
    # image_path = r"D:\Working\Images\Grayscale\Sample-4\BoardCheckSample-9.bmp"

    image_processor.set_image_path(image_path)

    # Preprocessing by resizing to 640x480
    resized_image_path, resized_image = image_processor.preprocessing(image_path, show_image=False)

    """
    Process and evaluate part 1 - Connector
    """
    # Check pin count
    if image_processor.check_connector_lock_defect(resized_image, 12, 1563, 1399, 202, 95):
        print("Connector Lock Defect Check: OK")
    else:
        print("Connector Lock Defect Check: NG")

    """
       Process and evaluate part 2 - Jack (FPC Lead)
    """
    image_processor.check_jack_fit_defect()
