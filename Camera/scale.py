import serial
import time
import re  # Regular expressions module

port = 'COM5'  # Replace with the correct COM port
baudrate = 9600  # Replace with the correct baud rate

def read_weight():
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            data = ser.readline().decode('ascii').strip()
            return data
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

def extract_weight(data):
    # Using regular expression to extract the weight
    match = re.search(r'(\d+\.\d+)', data)
    if match:
        return match.group(1)
    else:
        return None

while True:
    raw_data = read_weight()
    if raw_data:
        weight = extract_weight(raw_data)
        if weight:
            print(weight)
    time.sleep(0.025)  # Adjust the sleep time as needed for your application
