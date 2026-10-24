# Capture prompts for the Saluki hidden-state set: 12 per category, each with a
# tag (json01..json12, code01.., shell01.., math01.., prose01..). Plain ASCII.
# Each is wrapped with chat() from saluki_prompts.jl. None repeats those five.

const CAPTURE_PROMPTS = [
    (
        "json01",
        "Return only a JSON object describing a book with keys title, author, year, isbn and genres (an array of three strings). Use realistic values for a 1960s science fiction novel.",
    ),
    (
        "json02",
        "Return only a JSON array of five objects, each with id, name, price_usd and in_stock, describing items in a small hardware store's inventory.",
    ),
    (
        "json03",
        "Convert this YAML to JSON and return only the JSON:\n\nserver:\n  host: 0.0.0.0\n  port: 8080\n  tls: false\nlogging:\n  level: info\n  outputs:\n    - stdout\n    - /var/log/app.log\n",
    ),
    (
        "json04",
        "Return a JSON Schema (draft 2020-12) for a user profile with required fields email and display_name, an optional age integer from 13 to 120, and a list of string tags. Return only the schema.",
    ),
    (
        "json05",
        "Return only a JSON object for a calendar event: title 'Quarterly review', start 2026-11-03T14:00:00, end 2026-11-03T15:30:00, time zone Europe/Berlin, and attendees as a list of three email addresses (invent them under example.com).",
    ),
    (
        "json06",
        "Here is a JSON document with a mistake. Find the trailing comma and return the corrected JSON only:\n\n{\n  \"name\": \"sensor-7\",\n  \"readings\": [21.4, 21.9, 22.1,],\n  \"unit\": \"celsius\"\n}",
    ),
    (
        "json07",
        "Return only a JSON object that maps each of the seven days of the week to the number of hours a nurse works on a typical 3-day-on schedule. Use 12 for worked days and 0 for days off.",
    ),
    (
        "json08",
        "Produce a JSON object for an HTTP error response with status 404, a machine-readable code NOT_FOUND, a human message, and a details object that lists the missing resource type and id. Return only the JSON.",
    ),
    (
        "json09",
        "Flatten this nested JSON into dotted keys and return only the flat JSON object:\n\n{\"user\": {\"name\": \"Ada\", \"address\": {\"city\": \"Leeds\", \"zip\": \"LS1 4AP\"}}, \"active\": true}",
    ),
    (
        "json10",
        "Return only a JSON object with the fields of a Git commit: sha, author_name, author_email, date, subject and parents (a list). Invent plausible values.",
    ),
    (
        "json11",
        "I need a JSON config for a logging library with three named loggers: db, http and auth. Each has level and a list of handlers. Return only the JSON.",
    ),
    (
        "json12",
        "Return only a JSON object with two keys: primes, holding the first ten prime numbers as an array of integers, and sum, holding their total.",
    ),
    (
        "code01",
        "Write a Python function that returns the n-th Fibonacci number using iteration, and include two short test cases.",
    ),
    (
        "code02",
        "Implement a function in Go named ParseDuration that takes strings like 1h30m or 45s and returns the total number of seconds as an integer. Return only the code.",
    ),
    (
        "code03",
        "Write a JavaScript function debounce(fn, waitMs) that returns a debounced version of fn. Keep it under fifteen lines.",
    ),
    (
        "code04",
        "In Rust, write a function that takes a slice of i32 and returns the largest contiguous subarray sum. Explain the complexity in one sentence after the code.",
    ),
    (
        "code05",
        "Write a C function that reverses a null-terminated string in place and returns its length.",
    ),
    (
        "code06",
        "Write a SQL query that returns each customer's name and their total order amount for 2025, sorted from largest to smallest total. Tables: customers(id, name) and orders(id, customer_id, amount, created_at).",
    ),
    (
        "code07",
        "Write a Python class LRUCache with get and put methods and a capacity given at construction. Use collections.OrderedDict and keep it short.",
    ),
    (
        "code08",
        "Write a Java method that checks whether a string of brackets (), [], {} is balanced, and return a boolean.",
    ),
    (
        "code09",
        "Write a TypeScript type called DeepReadonly<T> that makes every nested property of an object type read-only, and show one usage example.",
    ),
    (
        "code10",
        "Write a Haskell function that merges two sorted lists into one sorted list.",
    ),
    (
        "code11",
        "Write a Lua function that splits a string on a given single-character delimiter and returns a table of pieces.",
    ),
    (
        "code12",
        "Write a small Ruby script that reads a CSV file named sales.csv with headers region and amount, and prints the total amount per region.",
    ),
    (
        "shell01",
        "Give me a one-line shell command that lists the ten largest files under the current directory, with sizes in human-readable form.",
    ),
    (
        "shell02",
        "Write a bash script that backs up the directory /etc/nginx into a dated tar.gz file in /var/backups, and deletes backups older than 14 days.",
    ),
    (
        "shell03",
        "How do I find every line containing the word timeout in all .log files under /var/log, ignoring case, and show the file name and line number? Give one command.",
    ),
    (
        "shell04",
        "Write a bash one-liner that counts how many times each IP address appears in the access log, most frequent first, assuming the IP is the first field.",
    ),
    (
        "shell05",
        "Explain what this command does, in plain words: find . -type f -mtime +30 -name '*.tmp' -print0 | xargs -0 rm -f",
    ),
    (
        "shell06",
        "Write a short bash script that waits until a TCP port 5432 on localhost accepts connections, trying once a second for at most 60 seconds.",
    ),
    (
        "shell07",
        "Give a git command sequence to create a new branch called fix-login from main, cherry-pick commit 4a2b9c1 onto it, and push it to origin with upstream tracking.",
    ),
    (
        "shell08",
        "Write a crontab line that runs /opt/scripts/sync.sh every 15 minutes on weekdays only, between 8 AM and 6 PM.",
    ),
    (
        "shell09",
        "Show a bash snippet that reads a file line by line and skips blank lines and lines starting with a hash sign.",
    ),
    (
        "shell10",
        "What is the difference between > file 2>&1 and &> file in bash? Show a small example of each.",
    ),
    (
        "shell11",
        "Write a PowerShell one-liner that lists all services that are running and whose names start with Win.",
    ),
    (
        "shell12",
        "Write a bash function that prints the free disk space percentage for a given mount point, using df.",
    ),
    (
        "math01",
        "A train leaves at 9:40 and travels 275 km at an average speed of 110 km/h. At what time does it arrive? Show the working.",
    ),
    ("math02", "Solve for x: 3(2x - 5) + 4 = 2(x + 7) - 1. Show each step."),
    (
        "math03",
        "A shop sells pens at 3 for 2.40 dollars. How much do 27 pens cost, and how many pens can you buy with 20 dollars? Show the working.",
    ),
    (
        "math04",
        "The sum of two numbers is 58 and their difference is 14. What are the numbers? Work through it with two equations.",
    ),
    (
        "math05",
        "A rectangle has a perimeter of 46 cm and its length is 5 cm more than its width. Find its area.",
    ),
    (
        "math06",
        "Compute the compound interest on 2000 dollars invested for 3 years at 4 percent per year, compounded annually. Give the final amount to the cent.",
    ),
    (
        "math07",
        "A tank fills in 6 hours with pipe A and in 9 hours with pipe B. How long do they take together? Show the rate reasoning.",
    ),
    (
        "math08",
        "What is the probability of rolling a total of 7 with two fair six-sided dice? Explain how you count the outcomes.",
    ),
    ("math09", "Expand and simplify (x + 3)(x - 4)(x + 1). Show the steps."),
    (
        "math10",
        "A car uses 6.5 litres of fuel per 100 km. How many litres are needed for a 740 km trip, and what does it cost at 1.85 per litre?",
    ),
    (
        "math11",
        "The average of five test scores is 78. Four of the scores are 72, 85, 90 and 64. What is the fifth score?",
    ),
    (
        "math12",
        "Find the slope and y-intercept of the line through the points (2, 7) and (6, 15), and write its equation.",
    ),
    (
        "prose01",
        "Explain to a high school student why the sky is blue, in about one paragraph.",
    ),
    (
        "prose02",
        "Write a short story of about 200 words about a retired bus driver who keeps finding notes left on the seat of her old route.",
    ),
    (
        "prose03",
        "Draft a polite email to my landlord asking when the broken radiator in the upstairs bedroom will be repaired. Keep it brief.",
    ),
    (
        "prose04",
        "Explain the difference between a virus and a bacterium to a curious twelve-year-old.",
    ),
    (
        "prose05",
        "Write a product description for a handmade ceramic coffee mug with a matte blue glaze. Two short paragraphs.",
    ),
    (
        "prose06",
        "Write a short dialogue between a grandmother and her grandson who are planting tomatoes for the first time.",
    ),
    (
        "prose07",
        "Summarize the causes of the French Revolution in a few sentences for a history exam review.",
    ),
    (
        "prose08",
        "Write the opening paragraph of a cover letter for a junior data analyst position at a regional hospital.",
    ),
    (
        "prose09",
        "Describe what a typical day looks like for a lighthouse keeper in the 1920s, written in the second person.",
    ),
    (
        "prose10",
        "Explain what inflation is and why central banks raise interest rates to fight it, in plain language.",
    ),
    (
        "prose11",
        "Write a friendly reminder email to a team that the quarterly planning meeting moves from Tuesday to Thursday at 10 AM.",
    ),
    (
        "prose12",
        "Write a short fable about a fox who learns that asking for help is not the same as being weak.",
    ),
]
