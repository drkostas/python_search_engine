# Python Search Engine
This is a search engine on the **Gutenberg Project* archive. 
It is implemented with python and the front end part is handled with **Flask** framework.

It is consisted of 2 basics elements:

The first one is the BuildIndex.py which scans the archive and creates an Inverted File and a file with the Document Names.
  I am using *3-in-4 Front Coding* compression.
  
The second part is the index.py(or QueryIndex.py as it was originally named). 
  Through the front-end part(form.html) the user enters the keywords. 
  The index.py is called which searches inside the inverted file using **tf*idf**.
  The results are separated in two parts: 
    All the documents the keywords found in and
    the documents in which the keywords were found as a phrase.
  Then the results are being printed firstly including all the documents and secondly the top 10 documents based on the **tf*idf**.

_**The link to the Front-End is in the Description.**_
