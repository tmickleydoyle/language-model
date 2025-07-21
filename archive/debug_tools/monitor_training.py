#!/usr/bin/env python3
"""Monitor training progress by checking log file and process status."""

import os
import time
import subprocess

def monitor_training():
    """Monitor the training process."""
    log_file = "training.log"
    
    print("🔍 TRAINING MONITOR")
    print("=" * 30)
    
    last_size = 0
    
    while True:
        try:
            # Check if training process is still running
            result = subprocess.run(
                ["ps", "aux"], 
                capture_output=True, 
                text=True
            )
            
            train_processes = [
                line for line in result.stdout.split('\n') 
                if 'train_streaming.py' in line and 'grep' not in line
            ]
            
            if not train_processes:
                print("❌ Training process not found")
                break
            
            # Get process info
            process_line = train_processes[0]
            parts = process_line.split()
            pid = parts[1]
            cpu = parts[2]
            mem = parts[3]
            
            print(f"📊 Process {pid}: CPU={cpu}%, Memory={mem}%")
            
            # Check log file size
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                if current_size > last_size:
                    print(f"📝 Log file grew: {current_size} bytes")
                    # Show new content
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        if new_content.strip():
                            print("📋 New output:")
                            print(new_content)
                    last_size = current_size
                else:
                    print(f"📝 Log file unchanged: {current_size} bytes")
            else:
                print("📝 No log file found")
            
            print("-" * 30)
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training()