# -*- coding: utf-8 -*-


def addTuples(tuple1, tuple2):
   return tuple([tuple1[i]+tuple2[i] for i in range(len(tuple1))])


def addListToList(list1, list2):
   for i in range(len(list1)):
      list1[i] += list2[i]


def subtractListFromList(list1, list2):
   for i in range(len(list1)):
      list1[i] -= list2[i]


def dotProduct(list1, list2):
   nReturn = 0
   for i in range(len(list1)):
      nReturn += list1[i] * list2[i]
   return nReturn


def addDictToDict(dict1, dict2):
   for key in dict2:
      if key in dict1:
         dict1[key] += dict2[key]
      else:
         dict1[key] = dict2[key]


def subtractDictFromDict(dict1, dict2):
   for key in dict2:
      if key in dict1:
         dict1[key] -= dict2[key]
      else:
         dict1[key] = -dict2[key]


def evaluateSentence(lCandidate, lReference):
   nCorrectWords = 0
   nCorrectTags = 0
   nChar = 0
   indexCandidate = 0
   indexReference = 0
   while lCandidate and lReference:
      if lCandidate[0][0] == lReference[0][0]:  # words right
         nCorrectWords += 1
         if lCandidate[0][1] == lReference[0][1]: # tags
            nCorrectTags += 1
         indexCandidate += len(lCandidate[0][0]) # move
         indexReference += len(lReference[0][0])
         lCandidate.pop(0)
         lReference.pop(0)
      else:
         if indexCandidate == indexReference:
            indexCandidate += len(lCandidate[0][0]) # move
            indexReference += len(lReference[0][0])
            lCandidate.pop(0)
            lReference.pop(0)
         elif indexCandidate < indexReference:
            indexCandidate += len(lCandidate[0][0])
            lCandidate.pop(0)
         elif indexCandidate > indexReference:
            indexReference += len(lReference[0][0]) # move
            lReference.pop(0)
   return nCorrectWords, nCorrectTags


def readNonEmptySentenceList(sents, bIgnoreNoneTag=True):
    out = []
    for sent in sents:
        lNewLine = []
        lLine = sent.split(' ')
        for nIndex in range(len(lLine)):
            tTagged = tuple(lLine[nIndex].split("_"))
            assert (len(tTagged) < 3)
            if len(tTagged) == 1:
                tTagged = (tTagged[0], "-NONE-")
            if (bIgnoreNoneTag == False) or (tTagged[0]):  # if we take -NONE- tag, or if we find that the tag is not -NONE-
                lNewLine.append(tTagged)
        out.append(lNewLine)
    return out


def score(sReference, sCandidate, verbose=False):
    nTotalCorrectWords = 0
    nTotalCorrectTags = 0
    nCandidateWords = 0
    nReferenceWords = 0
    reference = readNonEmptySentenceList(sReference)
    candidate = readNonEmptySentenceList(sCandidate)
    assert len(reference) == len(candidate)
    for lReference, lCandidate in zip(reference, candidate):
        n = len(lCandidate)
        nCandidateWords += len(lCandidate)
        nReferenceWords += len(lReference)
        nCorrectWords, nCorrectTags = evaluateSentence(lCandidate, lReference)
        nTotalCorrectWords += nCorrectWords
        nTotalCorrectTags += nCorrectTags
    word_precision = float(nTotalCorrectWords) / float(nCandidateWords)
    word_recall = float(nTotalCorrectWords) / float(nReferenceWords)
    tag_precision = float(nTotalCorrectTags) / float(nCandidateWords)
    tag_recall = float(nTotalCorrectTags) / float(nReferenceWords)
    if word_precision + word_recall > 0:
        word_fmeasure = (2 * word_precision * word_recall) / (word_precision + word_recall)
    else:
        word_fmeasure = 0.00001

    if tag_precision + tag_recall == 0:
        tag_fmeasure = 0.0
    else:
        tag_fmeasure = (2 * tag_precision * tag_recall) / (tag_precision + tag_recall)
    if verbose:
        return word_fmeasure, tag_fmeasure, word_precision, word_recall, tag_precision, tag_recall
        #return word_precision, word_recall, word_fmeasure, tag_precision, tag_recall, tag_fmeasure
    else:
        return word_fmeasure, tag_fmeasure