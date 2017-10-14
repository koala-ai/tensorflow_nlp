# -*- coding: utf-8 -*-

import sys, os
from nlp.segment.idcnn.dataset.sentence import Sentence


def process_token(token, sentence, out, end):
    nn = len(token)
    while nn > 0 and token[nn - 1] != '/':
        nn = nn - 1

    token = token[:nn - 1].strip()

    sentence.add_token(token)

    if end:
        x = [] # 保存char id
        y = [] # 保存char的标签
        sentence.generate_tr_line(x, y)
        for i in range(len(x)):
            line = x[i] + " " + y[i]
            out.write("%s\n" % line)
        out.write("\n")
        sentence.clear()

def processLine(line, out):
    line = line.strip()
    nn = len(line)
    seeLeftB = False
    start = 0
    sentence = Sentence()
    try:
        for i in range(nn):
            if line[i] == ' ':
                if not seeLeftB:
                    token = line[start:i]
                    if token.startswith('['):
                        tokenLen = len(token)
                        while tokenLen > 0 and token[tokenLen - 1] != ']':
                            tokenLen = tokenLen - 1
                        token = token[1:tokenLen - 1]
                        ss = token.split(' ')
                        for s in ss:
                            process_token(s, sentence, out, False)
                    else:
                        process_token(token, sentence, out, False)
                    start = i + 1
            elif line[i] == '[':
                seeLeftB = True
            elif line[i] == ']':
                seeLeftB = False
        if start < nn:
            token = line[start:]
            if token.startswith('['):
                tokenLen = len(token)
                while tokenLen > 0 and token[tokenLen - 1] != ']':
                    tokenLen = tokenLen - 1
                token = token[1:tokenLen - 1]
                ss = token.split(' ')
                ns = len(ss)
                for i in range(ns - 1):
                    process_token(ss[i], sentence, out, False)
                    process_token(ss[-1], sentence, out, True)
            else:
                process_token(token, sentence, out, True)
    except Exception as e:
        pass


# 生成训练和测试语料
def main(argc, argv):
    if argc < 3:
        print("Usage:%s <dir> <output>" % (argv[0]))
        sys.exit(1)
    rootDir = argv[1]
    out = open(argv[2], "w")
    for dirName, subdirList, fileList in os.walk(rootDir):
        for subdir in subdirList:
            level1Dir = os.path.join(dirName, subdir)
            print(level1Dir)
            for inDirName, inSubdirList, inFileList in os.walk(level1Dir):
                # curDir = os.path.join(level1Dir, inDirName)
                for file in inFileList:
                    if file.endswith(".txt"):
                        cur_file = os.path.join(level1Dir, file)
                        print("File name: %s" % cur_file)
                        fp = open(cur_file, "r")
                        for line in fp.readlines():
                            line = line.strip()
                            processLine(line, out)
                        fp.close()
    out.close()


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)