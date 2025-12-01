import sys

def compress(seq):
    # Convert each base64 character list to a string and format as vector
    result = "{"
    for i, b in enumerate(seq):
        if i > 0:
            result += ", "
        result += '"' + "".join(b) + '"'
    result += "}"
    return result

def decompress(data):
    # Not needed for vector output format
    pass

if __name__ == "__main__":
    sys.stdin = open("in", "rt")
    sys.stdout = open("out", "wt")

    mod = 10**9+7
    seq = []
    # Base64 characters
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    
    for _ in range(10000):
        n, x = map(int, input().split())
        val = x % mod
        base64_chars = []
        tmp = val
        while tmp > 0:
            base64_chars.append(chars[tmp % 64])
            tmp //= 64
        if not base64_chars:
            base64_chars = ['A']  # 'A' represents 0
        base64_chars = base64_chars[::-1]
        seq.append(base64_chars)

    # Output as vector of strings
    compressed = compress(seq)
    sys.stdout.write(compressed)
