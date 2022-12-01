import requests
import re
import tkinter
import tkinter.font


def tick1Min():
    url = 'http://www.kma.go.kr/wid/queryDFSRSS.jsp?zone=4139054000'
    response = requests.get(url)

    temp = re.findall(r'<temp>(.+)</temp>', response.text)
    humi = re.findall(r'<reh>(.+)</reh>', response.text)
    display = str(temp[0]) + "C" + " " + str(humi[0]) + "%"

    label.config(text=display)
    window.after(60000, tick1Min)
    return [temp, humi]


window = tkinter.Tk()
window.title("TEMP HUMI DISPLAY")
window.geometry("400x100")
window.resizable(False, False)

font = tkinter.font.Font(size = 30)
label = tkinter.Label(window, text=" ", font=font)
label.pack()

a= tick1Min()
temp = a[0]
humi = a[1]
temp

window.mainloop()







