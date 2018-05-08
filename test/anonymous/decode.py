import support

def decode(ciphertext, out_file):
    g = open('data.txt','r')
    test = g.readlines()
    print test
    g.close()
    support.main()
    print "in decode"
    s = ciphertext[:20]
    f = open(out_file, 'w')
    f.write(s)
    f.close()

