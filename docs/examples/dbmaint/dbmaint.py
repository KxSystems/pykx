import os
from pathlib import Path
import shutil
import subprocess

from pykx import q


def setup():
    os.chdir(Path(__file__).resolve().parent)
    try:
        shutil.rmtree('hdb')
    except FileNotFoundError:
        pass
    subprocess.run(('q', 'setup_hdb.q', '-db', 'hdb'))


def demo():
    dbdir = Path('hdb').resolve(strict=True)

    # Rename the `px column to `price
    q.dbmaint.renamecol(dbdir, 'trades', 'px', 'price')

    # Add the column `industry with the default value set to "tech"
    q.dbmaint.addcol(dbdir, 'trades', 'industry', [b'tech'])

    # Convert `industry from strings to symbols (enumerated over `sym)
    fn = q('{(` sv x,`sym)?`$y}', dbdir) # create a projection
    q.dbmaint.fncol(dbdir, 'trades', 'industry', fn)

    # Reverse the order of the columns
    q.dbmaint.reordercols(
        dbdir, 'trades', list(reversed(q.dbmaint.listcols(dbdir, 'trades'))))

    # Casting the `price column to reals
    q.dbmaint.castcol(dbdir, 'trades', 'price', 'real')


if __name__ == '__main__':
    setup()
    demo()
