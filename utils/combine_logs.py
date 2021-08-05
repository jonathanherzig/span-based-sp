"""Prints all available logs."""

import os

if __name__ == '__main__':
    path_to_logs = 'logs/'
    header = None
    body = []
    for filename in os.listdir(path_to_logs):
        if filename.startswith('log'):
            with open(os.path.join(path_to_logs, filename), 'r') as f:
                lines = f.readlines()
                if header is None:
                    header = lines[0]
                for i, line in enumerate(lines):
                    if i > 0:
                        body.append(line)
    body.sort()
    print(header.strip())
    for l in body:
        print(l.strip())
