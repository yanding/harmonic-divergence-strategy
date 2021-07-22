# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
print("Hello world")
values = [1,4,2,5,3,7,2,5,6,1,5]
curr = values[5]
window = 5
is_greater = True
for index in range(0,window):
    left = values[index]
    right = values[window - index]
    if curr < left or curr < right:
        is_greater = False
        
print(is_greater)