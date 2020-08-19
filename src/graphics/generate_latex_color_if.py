


row = '\ifnum\Current=!val! \n \\node[fill={rgb,255:red,!c!; green,!c!; blue,!c!}, minimum width=10mm, minimum height=10mm] at (\X-0.5,-\Y+0.5) {\\rotatebox[origin=c]{\d}{$\\rightarrow$}}; \n \\fi \n'

x = row.split('!val!')

with open('temp.txt', 'w') as file:
    
    # change to this exp color grade 1.1^(-x^(2/3)) - 0.13
    for i in range(100):
        new_row = x[0] + str(i)
        new_row += x[1].split('!c!')[0]
        new_row += str(int(255-i)) # RED
        new_row += x[1].split('!c!')[1]
        new_row += str(int(255-(i*2))) # GREEN
        new_row += x[1].split('!c!')[2]
        new_row += str(int(255-(i*1.7))) # BLUE
        new_row += x[1].split('!c!')[3]
        if i == 0: 
            new_row = new_row.split('$\\rightarrow$')[0] + new_row.split('$\\rightarrow$')[1]
        file.write(new_row)
