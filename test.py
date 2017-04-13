#!C:\Program Files\Python35\python.exe
from os import *  
import cgi  
form = cgi.FieldStorage()

print ("""
		<div>
		<br>result 1
		<br>result 2
		<br>result 3
		<br>result 4
		<br>""",form["Query"].value,"""
		</div>
		""")