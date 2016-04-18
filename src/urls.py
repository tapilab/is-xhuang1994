import urllib.request
import urllib.error


f = open("urls/urls_polluters.txt", 'r')
f1 = open("urls/urls.txt", 'w')
f2 = open("urls/urls_not_found.txt", 'w')
f3 = open("urls/urls_forbidden.txt", 'w')
f4 = open("urls/uerrors.txt", 'w')
f5 = open("urls/unicode_errors.txt", 'w')
f6 = open("urls/other_errors.txt", 'w')
for line in f:
    line = line[:len(line)]
    try:
        with urllib.request.urlopen(line) as response:
            f1.write(line)
            f1.flush()
    except urllib.error.HTTPError as herror:
        if herror.code == 404:
            f2.write(line)
            f2.flush()
        elif herror.code == 403:
            f3.write(line)
            f3.flush()
        else:
            print(line)
            print(herror.code, herror.reason)
    except urllib.error.URLError as uerror:
        f4.write(line)
        f4.write(uerror.__str__())
        f4.flush()
    except UnicodeError:
        f5.write(line)
        f5.flush()
    except Exception as e:
        f6.write(line)
        f6.write(e.__str__())
        f6.flush()

f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()