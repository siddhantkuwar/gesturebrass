def main():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv2.imshow("GestureBrass", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
