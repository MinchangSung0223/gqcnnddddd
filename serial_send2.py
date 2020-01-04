import serial
ser = serial.Serial('/dev/ttyUSB1', 2400, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)

print(ser)
left = bytearray([255, 1, 0, 4, 35, 0, 40])
right = bytearray([255, 1, 0, 2, 35, 0, 38])
up = bytearray([0xff, 0x01, 0, 0x08, 0x00, 0x23, 0x2c])
down = bytearray([0xff, 0x01, 0x00, 0x10, 0x00, 0x20, 0x31])
stop = bytearray([0xff, 0x01, 0x00, 0x00, 0x0, 0x00, 0x01])
values = stop
while 1:
   print("INPUT : ")
   a = int(input())
   if a == 1 :
      values = left
   elif a == 2:
      values = right
   elif a == 3:
      values = up
   elif a == 4:
      values = down
   elif a == 5:
      values = stop
   print(values)
   ser.write(values)
ser.close()

