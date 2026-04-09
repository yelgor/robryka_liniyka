import cv2
from src.cv.pipeline import CVPipeline
from src.models.frame_packet import FramePacket
import time

image = cv2.imread('assets/photo_2026-04-07_23-54-14.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# cv2.imshow("RGB Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

frame_packet = FramePacket(frame_id=0, image=image_rgb, timestamp=time.time())

pipeline = CVPipeline()

result = pipeline.process(frame_packet)

debug_image_bgr = cv2.cvtColor(result.debug_image, cv2.COLOR_RGB2BGR)

print("target_found:", result.target_found)
print("bbox:", result.bbox)

cv2.imshow("debug", debug_image_bgr)
cv2.waitKey(0)
# cv2.destroyWindow("debug")

