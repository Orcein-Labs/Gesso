# Chat prompts shared by the Saluki CPU scripts (thinking off).

const PROMPTS = [
    "json" => "Return only a JSON object for a tool call to get_weather with arguments city set to Toronto, units set to metric and days set to 3. No prose.",
    "code" => "Rewrite this function so it skips empty lines, and print the whole file back:\n\n```python\ndef count_words(path):\n    total = 0\n    with open(path) as f:\n        for line in f:\n            total += len(line.split())\n    return total\n\n\ndef main():\n    import sys\n    print(count_words(sys.argv[1]))\n\n\nif __name__ == \"__main__\":\n    main()\n```",
    "shell" => "Summarize this command output in two sentences:\n\n```\n\$ df -h\nFilesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2  476G  431G   41G  92% /\n/dev/sda1       3.6T  2.9T  700G  81% /mnt/d\ntmpfs            16G  1.2G   15G   8% /tmp\n```",
    "math" => "Compute 17 * 23 + 144 / 12 - 9 step by step, then give the final number.",
    "prose" => "Write a short paragraph about a lighthouse keeper who finds a letter in a bottle.",
]

chat(msg) = "<|im_start|>user\n" * msg * "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
