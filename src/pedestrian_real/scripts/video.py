import cv2
def catch_video(name='my_video', video_index=0):
    # cv2.namedWindow(name)
    cap = cv2.VideoCapture(video_index) # 创建摄像头识别类
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(name, 2048, 1024)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        # 如果没有检测到摄像头，报错
        raise Exception('Check if the camera is on.')
    while cap.isOpened():        
        catch, frame = cap.read()  # 读取每一帧图片
        cv2.imshow(name, frame) # 在window上显示图片
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            # 按q退出
            break
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":    
    catch_video()