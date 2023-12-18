#!/usr/bin/env python3
"""Create the loop"""

while True:
    user_input = input('Q: ')
    if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
        print('A: Goodbye')
        break
    else:
        print('A: ')
