# from RPLCD.i2c import CharLCD
import time 
import socket


lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2)


lcd.clear()  
lcd.cursor_mode = 'blink' 


lcd.write_string("Hello, Lil!")



def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        print(f"Error retrieving IP address: {e}")
        return "No IP"




lcd.clear()
lcd.write_string("IP Address:")  # Line 1
lcd.crlf()                       # Move to Line 2
lcd.write_string(get_ip_address())  # Display the IP address


