import argparse
import os
from os.path import exists
from stat import ST_SIZE
from time import localtime, strftime, time

program = "../llama-np.py"
output_dir = "output"
ref_dir = "reference"
model_dir = "/home/yuan/DLModels/lumi/"

# Each option has an id, which is the last element of each tuple.
# The id is used to uniquely identify a test and used to name the test results.
# You can comment out the tests. Avoid changing the id.

# (model_file, seqlength_scale, id)
# Because of the different tokenizer used, the numbers of tokens are different.
llama2_models = [
    (model_dir + "stories260k.lmw", 3.0, "m21"),
    (model_dir + "stories15m.lmw", 1.2, "m22"),
    (model_dir + "stories42m.lmw", 1.2, "m23"),
    (model_dir + "stories110m.lmw", 1.2, "m24"),
    (model_dir + "tinyllama-1.1b-3t.lmw", 1.2, "m25"),
    (model_dir + "tinyllama-1.1b-chat-v1.0.lmw", 1.2, "m26"),
    (model_dir + "llama-2-7b.lmw", 1.2, "m27"),
    (model_dir + "llama-2-7b-chat.lmw", 1.2, "m28"),
]
llama3_models = [
    (model_dir + "llama-3-8b.lmw", 1.2, "m31"),
    (model_dir + "llama-3-8b-instruct.lmw", 1.2, "m32"),
    (model_dir + "qwen1.0-7b-chat.lmw", 1.2, "m33"),

]

# (prompt, seqlength, id)
prompts = [
    ("-i 'It is easy'", 16, "p1"),
    (
        "-i 'There are three red balls and four blue balls in a bag. If I take one ball'",
        32,
        "p2",
    ),
    ("-i 'List all the prime numbers between 100 and 200. They are'", 32, "p3"),
    ("-i '你好'", 24, "p4"),
    ("-i 'The US president in 2020 is'", 24, "p5"),
    ("-f input_prompt1_short.txt", 20, "p6"),
    ("-f input_prompt2_long.txt", 800, "p7"),
]

cupy_options = [
    ("", "c0"),
    ("--cupy", "c1"),
]

kv_cache_options = [
    ("", "k0"),
    ("--nokvcache", "k1"),
    ("--useinplacekvcache", "k2"),
]

prefill_options = [
    ("", "f0"),
    ("--fill1", "f1"),
]

sampler_options = [
    ("--temp 0", "s1"),
    ("--temp 0.4", "s2"),
    ("--topp 0.99", "s3"),
]

output_options = [
    ("--emit-one-token", "o1"),
]


def find_model(str):
    for m in llama2_models + llama3_models:
        if str in m[0]:
            return m
    return None


def iterate_all(func, ccheck=None):
    all_models = llama2_models + llama3_models
    for m in all_models:
        for p in prompts:
            for c in cupy_options:
                func(
                    f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {c[0]}",
                    m[-1] + p[-1] + c[-1],
                )

    # check kv_cache_options and prefill_options for llama2 models
    # prefill is very slow for long prompts, so we only test it for a shorter prompt.
    for m in [find_model("stories110m.lmw"), find_model("stories260k.lmw")]:
        for p in [prompts[1]]:
            for c in cupy_options:
                for k in kv_cache_options[1:]:
                    for f in prefill_options[1:]:
                        func(
                            f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {c[0]} {k[0]} {f[0]}",
                            m[-1] + p[-1] + c[-1] + k[-1] + f[-1],
                        )

    iterate_cross(func, ccheck)


def iterate_cross(func, ccheck=None):
    # check kv_cache_options and prefill_option for llama3 model
    # Use the output_options (emit-one-token), so we can cross-compare all the outputs.
    m = llama3_models[1]
    p = prompts[1]
    o = output_options[0]
    all_outputs = []
    for c in cupy_options:
        for k in kv_cache_options:
            for f in prefill_options:
                result, filename = func(
                    f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {c[0]} {k[0]} {f[0]} {o[0]}",
                    m[-1] + p[-1] + c[-1] + k[-1] + f[-1] + o[-1],
                )
                all_outputs.append(filename)

    if ccheck is not None:
        ccheck(all_outputs)


def iterate_l0(func, ccheck=None):
    for m in [find_model("stories15m.lmw"), find_model("llama-3-8b-instruct.lmw")]:
        for p in [prompts[1]]:
            for c in cupy_options:
                func(
                    f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {c[0]}",
                    m[-1] + p[-1] + c[-1],
                )
    for m in [find_model("stories15m.lmw")]:
        for p in [prompts[-1]]:
            for c in cupy_options:
                func(
                    f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {c[0]}",
                    m[-1] + p[-1] + c[-1],
                )


def iterate_l1(func, ccheck=None):
#    iterate_l0(func, ccheck)

    '''
    for m in [find_model("stories15m.lmw"), find_model("llama-3-8b-instruct.lmw")]:
        for p in [prompts[1]]:
            for c in cupy_options:
                for k in kv_cache_options:
                    func(
                        f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {c[0]} {k[0]}",
                        m[-1] + p[-1] + c[-1] + k[-1],
                    )

    for m in [find_model("llama-3-8b-instruct.lmw")]:
        for p in [prompts[-1]]:
            for c in cupy_options:
                func(
                    f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {c[0]}",
                    m[-1] + p[-1] + c[-1],
                )
    '''

    for m in [find_model("qw")]:
        for p in [prompts[3]]:
            for c in cupy_options:
                func(
                    f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {c[0]}",
                    m[-1] + p[-1] + c[-1],
                )

    '''
    m = find_model("llama-3-8b-instruct.lmw") 
    p = prompts[0]
    for o in output_options:
        for s in sampler_options:
            func(
                f"-w {m[0]} {p[0]} --seqlength {int(m[1]*p[1])} {o[0]} {s[0]}",
                m[-1] + p[-1] + o[-1] + s[-1],
            )
    '''



def show(str1, str2):
    print(str1)
    print(str2)


total_number_of_tests = 0


def count_test(*args):
    global total_number_of_tests
    total_number_of_tests += 1
    return 0, ""


current = 0
passed = []
failed = []
miscompare = []
ref_missing = []


def test_one(cmd, ids):
    global current, passed, failed
    current = current + 1
    output = output_dir + "/" + ids
    command = f"{cmd} --seed 34 >{output}"

    exists = None
    if os.path.exists(output + ".pass"):
        exists = "pass"
        compare_result = 0
        passed.append((command, ids))
    elif os.path.exists(output + ".fail"):
        exists = "fail"
        compare_result = 3
        failed.append((command, ids))
    elif os.path.exists(output + ".misc"):
        exists = "misc"
        compare_result = 1
        miscompare.append((command, ids))

    if exists is not None:
        #skip the test
        text = f"[{current}/{total_number_of_tests}]:old {exists} {command}"
        print(text)
        return compare_result, output+"."+exists

    status = "Testing"
    local_time1 = strftime("%H:%M:%S", localtime())
    text = f"[{current}/{total_number_of_tests}]:{status} {local_time1} {command}"
    print(text, end="", flush=True)

    start_time = time()
    result = os.system(f"python {program} {command} 2>&1")
    end_time = time()

    execution_time = end_time - start_time
    minutes = int(execution_time / 60)
    seconds = int(execution_time % 60)

    # 91 red, 92 green, 93 yellow
    if result != 0:
        status = "\033[91m Failed\033[00m"
        failed.append((command, ids))
        filename = output + ".fail"
        os.rename(output, filename)
        compare_result = 3
    else:
        compare_result = compare(output, ref_dir + "/" + ids + ".pass")
        if compare_result == 1:
            status = "\033[93m Miscompare\033[00m"
            miscompare.append((command, ids))
            filename = output + ".misc"
            os.rename(output, filename)
        elif compare_result == 0:
            status = "\033[92m Passed\033[00m"
            passed.append((command, ids))
            filename = output + ".pass"
            os.rename(output, filename)
        else:
            status = "\033[93m Ref Missing\033[00m"
            ref_missing.append((command, ids))
            filename = output + ".pass"
            os.rename(output, filename)

    text = f"\r[{current}/{total_number_of_tests}]:{status} {local_time1} {minutes}m{seconds}s {command}"
    print(text)
    return compare_result, filename
    # 0 passed, 1 miscompare, 2 ref missing, 3 failed


cross_failed = None


def cross_check(all_outputs):
    failed = 0
    global cross_failed
    cross_failed = []
    # compare all the outputs pairwise
    for i in range(0, len(all_outputs)-1):
        for j in range(i + 1, len(all_outputs)):
            compare_result = compare(
                all_outputs[i], all_outputs[j],
            )
            if compare_result == 0:
                print(f"Cross compare passed: {all_outputs[i]} {all_outputs[j]}")
            else:
                print(f"Cross compare failed: {all_outputs[i]} {all_outputs[j]}")
                failed += 1
                cross_failed.append((all_outputs[i], all_outputs[j]))
    return failed


def compare(str1, str2):
    """
    Compare two files str1 and str2. Both are text files.
    If they are the same, return True.
    If they are different, check their differences line by line.
    If the line that is different contains the string "seconds", ignore the difference.
    Return False if the lines are different otherwise.

    Return value
     0: same
     1: different
     2: missing reference file
    """
    if not os.path.exists(str2):
        return 2

    with open(str1, "r") as file1, open(str2, "r") as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
        if lines1 == lines2:
            return 0
        else:
            for line1, line2 in zip(lines1, lines2):
                if "seconds" in line1 and "seconds" in line2:
                    continue
                if "tok/s" in line1 and "tok/s" in line2:
                    continue
                if line1 != line2:
                    return 1
            return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a", action="store_true", help="all")
    group.add_argument(
        "-l0",
        action="store_true",
        help="l0: sanity check, finish in less than 1 minute.",
    )
    group.add_argument(
        "-lc",
        action="store_true",
        help="lc: cross check. Check options that do not affect generated results.",
    )
    group.add_argument(
        "-l1",
        action="store_true",
        default=False,
        help="l1: fast check, finish in less than 10 minutes.",
    )

    parser.add_argument("-noclear", action="store_true", help="Do not clear the output directory.")

    args = parser.parse_args()

    if args.a:
        iterate_func = iterate_all
        test_name = "all"
    elif args.l0:
        iterate_func = iterate_l0
        test_name = "l0"
    elif args.l1:
        iterate_func = iterate_l1
        test_name = "l1"
    elif args.lc:
        iterate_func = iterate_cross
        test_name = "lc"

    iterate_func(count_test)
    print(f"Test kind: {test_name}  Total number of tests: {total_number_of_tests}")

    start_time = time()

    if not args.noclear:
        os.system(f"rm {output_dir}/*")

    current = 0
    iterate_func(test_one, cross_check)
    print(f"passed: {len(passed)}")
    print(f"failed: {len(failed)}")
    print(f"miscompare: {len(miscompare)}")
    print(f"ref missing: {len(ref_missing)}")
    if cross_failed is not None:
        print(f"cross failed: {len(cross_failed)}")
    else:
        print(f"cross failed: n/a")

    end_time = time()
    execution_time = end_time - start_time
    minutes = int(execution_time / 60)
    seconds = int(execution_time % 60)

    print(f"Execution time: {minutes} minutes {seconds} seconds")
