import sys
from dump import dump
from ensight import ensight
d = dump("dump.peri");
d.map(1,"id",2,"type",3,"x",4,"y",5,"z",6,"damage");
e = ensight(d);
e.one("disk","damage","Damage")
