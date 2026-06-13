import cv2
import subprocess
import json
import os
import sys

def main():
    # Ensure the Rust CLI is built
    print("Building Rust VIGA CLI...")
    subprocess.run(["cargo", "build", "--release"], check=True)
    
    cli_path = "target/release/viga_cli"
    if not os.path.exists(cli_path):
        print(f"Error: {cli_path} not found.")
        sys.exit(1)
        
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    # Start Rust VIGA CLI as a subprocess
    process = subprocess.Popen(
        [cli_path], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        text=True,
        bufsize=1 # Line buffered
    )
    
    tmp_frame_path = "tests/test_data/live_frame.jpg"
    
    print("Live demo running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Write frame to disk for Rust to process
        cv2.imwrite(tmp_frame_path, frame)
        
        # Send path to Rust CLI
        process.stdin.write(tmp_frame_path + "\n")
        process.stdin.flush()
        
        # Read alerts JSON back
        line = process.stdout.readline()
        if not line:
            break
            
        try:
            alerts = json.loads(line)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {line}")
            alerts = []
            
        # Draw alerts
        if len(alerts) > 0:
            # Draw a big warning banner
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, "SAFETY ALERT: MISSING GEAR DETECTED", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            for alert in alerts:
                # Bounding box is [x, y, w, h]
                box = alert["bounding_box"]
                x = int(box["x"])
                y = int(box["y"])
                w = int(box["width"])
                h = int(box["height"])
                
                # Draw red bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
                # Draw alert text
                cv2.putText(frame, f"Person {alert['person_id']}: Missing {alert['missing_gear']}", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                # Draw a hint of the geometric ray
                # (For visual flair, we just draw a line from bottom of box towards center)
                cv2.line(frame, (x + w//2, y + h), (int(box["x"] + w//2), int(box["y"] + h + 100)), (0, 165, 255), 2)
        else:
            # Draw safe banner
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 255, 0), -1)
            cv2.putText(frame, "ALL CLEAR", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imshow("VIGA Safety Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    process.terminate()

if __name__ == "__main__":
    main()
