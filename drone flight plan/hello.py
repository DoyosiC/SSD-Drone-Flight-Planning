from djitellopy import Tello
import cv2
import time
import keyboard


tello = Tello()
tello.connect()
time.sleep(0.2)
tello.streamon()
print("------------------conntected OK-------------------")
time.sleep(1.0)

try:
    frame_read =  tello.get_frame_read()
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        image = frame_read.frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480,360))
        cv2.imshow("drone", image)

        if keyboard.is_pressed("q"):  # qキーが押されたら離脱
            tello.end()
            time.sleep(0.1)
            cv2.destroyAllWindows()
            break
        elif keyboard.is_pressed("t"):  # tキーが押されたら離陸
            tello.takeoff()
            time.sleep(0.2)
        elif keyboard.is_pressed("g"):  # gキーが押されたら着陸
            tello.land()
            time.sleep(0.2)
        elif keyboard.is_pressed("w"):  # wキーが押されたら前進
            tello.send_rc_control(0, 30, 0, 0)
        elif keyboard.is_pressed("s"):  # sキーが押されたら後退
            tello.send_rc_control(0, -30, 0, 0)
        elif keyboard.is_pressed("a"):  # aキーが押されたら左移動
            tello.send_rc_control(-30, 0, 0, 0)
        elif keyboard.is_pressed("d"):  # dキーが押されたら右移動
            tello.send_rc_control(30, 0, 0, 0)
        else:
            tello.send_rc_control(0, 0, 0, 0)


except (KeyboardInterrupt, SystemExit,Exception) as e:    # Ctrl+cが押されたら離脱
    tello.end()
    time.sleep(0.1)
    cv2.destroyAllWindows()
    print(f"error : {e}")