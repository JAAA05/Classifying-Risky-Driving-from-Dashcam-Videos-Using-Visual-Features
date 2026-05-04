import cv2
import numpy as np

def enhance_low_light(image):
    """
    Enhances low-light images using CLAHE in the LAB color space.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge and convert back to BGR
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Optional: Gamma correction to further brighten mid-tones
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    
    return enhanced

class OpenCVLaneDetector:
    def __init__(self, width=640, height=640):
        self.width = width
        self.height = height
        
    def get_lane_mask(self, frame):
        """
        Detects lanes using traditional CV: HSL filtering + Canny + ROI.
        Returns a binary mask of the same size as the input frame.
        """
        # 1. Pre-process
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. HSL filtering for white/yellow lines (robust to lighting)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        # White mask
        lower_white = np.uint8([0, 200, 0])
        upper_white = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(hls, lower_white, upper_white)
        # Yellow mask
        lower_yellow = np.uint8([10, 0, 100])
        upper_yellow = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
        
        combined_binary = cv2.bitwise_or(white_mask, yellow_mask)
        
        # 3. Edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. Combine HSL and Edges
        final_mask = cv2.bitwise_and(edges, combined_binary)
        
        # 5. Region of Interest (Triangle/Trapezoid at bottom)
        roi_mask = np.zeros_like(final_mask)
        h, w = final_mask.shape
        polygon = np.array([[
            (0, h),
            (w // 2 - 50, h // 2 + 50),
            (w // 2 + 50, h // 2 + 50),
            (w, h)
        ]], np.int32)
        cv2.fillPoly(roi_mask, polygon, 255)
        masked_edges = cv2.bitwise_and(final_mask, roi_mask)
        
        # 6. Hough Transform to "clean" the lines
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=150)
        
        hough_mask = np.zeros_like(final_mask)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 10) # Thick lines to help overlap
                
        return hough_mask

def ensemble_lane_detector(yolop_mask, opencv_mask):
    """
    Combines YOLOP and OpenCV masks. 
    Logic: If YOLOP fails (low pixel count), trust OpenCV. 
    Otherwise, take a weighted average or bitwise intersection/union.
    """
    yolop_weight = 0.7
    cv_weight = 0.3
    
    # Check if YOLOP found anything significant
    yolop_pixels = np.count_nonzero(yolop_mask)
    if yolop_pixels < 500: # Heuristic threshold
        return opencv_mask
    
    # Simple weighted union
    combined = (yolop_mask.astype(float) * yolop_weight + 
                opencv_mask.astype(float) * cv_weight)
    
    return (combined > 0.4).astype(np.uint8)
