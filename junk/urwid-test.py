#!/usr/bin/python

#
# Text
# Edit
# Filler
# SimpleListWalker  <-
# ListBox
# MainLoop
# Divider
# Pile
#
# AttrMap
# AttrWrap (deprecated)
#
#
#


import sys
import os
import subprocess

import urwid

palette = [
        ('banner', 'black', 'light gray', 'standout,underline'),
        ('header', 'yellow', 'dark blue', 'bold,standout,underline'),
        ('header2', 'yellow', 'dark red', 'bold,standout,underline'),
        ('streak', 'black', 'dark red', 'standout'),
        ('fm', 'white', 'dark blue', 'standout'),
        ('key', 'yellow', 'black', 'standout'),
        ('bg', 'black', 'dark blue'),]

def ui(input):
    if input == 'ctrl e':
        kolumny.set_focus(0)
        return
    elif input == 'ctrl r':
        kolumny.widget_list[0] = urwid.AttrMap(FM(urwid.TreeWalker(DirectoryNode('..'))), 'fm')
    elif input == 'f6':
        global subprocesses
        for p in subprocesses:
            os.kill(p.pid, 9)
            main_loop.remove_watch_file(p.stdout.fileno())
        subprocesses = []
        footer1.set_text(footer1_txt + str(subprocesses))
    elif input == 'enter':
        txt.set_text(('header', 'value: ' + ask.edit_text))
        ask.edit_text = ''
        return
    txt.set_text(('header', 'BLA: ' + input))

txt = urwid.Text(('header', "   Ctrl-R -- quit   "), align="left")
ask = urwid.Edit(('banner', "Objective value:\n"))

content = urwid.SimpleListWalker([txt,
    urwid.Divider('-'),
    ask,
    urwid.Divider('-'),
    urwid.Text(open('/etc/passwd').read(), wrap='clip'),
    urwid.Divider('-'),
    ])
listbox = urwid.ListBox(content)

head = urwid.AttrMap(urwid.Text('QOpt :: Quantum-Inspired Evolutionary Algorithms framework', align='center', wrap='clip'), 'header2')

class TreeNodeWidget(urwid.TreeWidget):
    def __init__(self, node):
        urwid.TreeWidget.__init__(self, node)
        self.expanded = node.get_depth() == 0
        #self.update_expanded_icon()
        self._w = urwid.AttrWrap(self._w, None)
        self._w.attr = 'banner'
        self._w.focus_attr = 'header2'
    def get_display_text(self):
        if self.get_node().get_depth() == 0:
            n = 'QOPT'
        else:
            n = os.path.basename(self.get_node().get_key())
        if self.get_node().__class__ == DirectoryNode:
            return '[%s]' % n
        else:
            return n
    def selectable(self):
        return True
    def keypress(self, size, key):
        key = self.__super.keypress(size, key)
        if key == 'v':
            f = fm.get_focus()[1].get_key()
            if not os.path.isdir(f):
                s = open(f).read()
                output.set_edit_text(s)
        if key == 'enter':
            if self.is_leaf:
                f = fm.get_focus()[1].get_key()
                if f.endswith('.py'):
                    p = subprocess.Popen(["python", f], shell=False,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    subprocesses.append(p)
                    footer1.set_text(footer1_txt + str(subprocesses))
                    def cb():
                        output.set_edit_text(output.get_edit_text() + p.stdout.readline())
                    main_loop.watch_file(p.stdout.fileno(), cb)
                    #output.set_edit_text(p.stdout.readline())
                    #os.system('python %s' % f)
                else:
                    s = open(f).read()
                    output.set_edit_text(s)
            else:
                self.expanded = not self.expanded
                self.update_expanded_icon()
        return key


class FileNode(urwid.TreeNode):
    def __init__(self, value, parent = None, depth = None):
        urwid.TreeNode.__init__(self, value, key = value, parent = parent, depth = depth)
    def load_widget(self):
        return TreeNodeWidget(self)

class DirectoryNode(urwid.ParentNode):
    # key -> path
    def __init__(self, value, parent = None, depth = 0):
        urwid.ParentNode.__init__(self, value, key = value, parent = parent, depth = depth)
    def load_child_keys(self):
        k = self.get_key() + '/'
        c = sorted(filter(lambda f: os.path.isdir(k+f), os.listdir(k))) + \
                sorted(filter(lambda f: os.path.isfile(k+f), os.listdir(k)))
        c = map(lambda f: os.path.join(self.get_key(), f), c)
        return c
    def load_child_node(self, key):
        if os.path.isdir(key):
            return DirectoryNode(key, parent = self, depth = self.get_depth() + 1)
        return FileNode(key, parent = self, depth = self.get_depth() + 1)
    def load_widget(self):
        return TreeNodeWidget(self)

class FM(urwid.TreeListBox):
    def keypress(self, size, key):
        if key == 'j':
            key = 'down'
        elif key == 'k':
            key = 'up'
        elif key == 'l':
            key = 'right'
        elif key == 'J':
            key = 'end'
        elif key == 'K':
            key = 'home'
        elif key == 'h':
            key = 'left'
        elif key == 'e':
            os.system('vim "%s"' % str(self.get_focus()[1].get_key()))
            main_loop.draw_screen()
        elif key == 'E':
            os.system('xterm -e \'vim "%s"\'' % str(self.get_focus()[1].get_key()))
        self.__super.keypress(size, key)
        return key


subprocesses = []

fm = FM(urwid.TreeWalker(DirectoryNode('..')))
output = urwid.Edit('', wrap='clip', multiline=True)

kolumny = urwid.Columns([
    ('fixed', 30, urwid.AttrMap(fm, 'fm')),
    ('fixed', 1, urwid.Filler(urwid.Text('|' * 500), 'top')),
    ('weight', 3, urwid.ListBox(urwid.SimpleListWalker([
        urwid.Text('Output:'),
        urwid.Divider('-'),
        output,
        urwid.Divider('-'),
        ])) ),
    ], focus_column=0, dividechars=1)


footer1_txt = 'Subprocesses: '
footer1 = urwid.Text('')
footer1.set_text(footer1_txt + str(subprocesses))

footer = urwid.Pile([
    urwid.AttrMap(footer1, 'header2'),
    urwid.AttrMap(urwid.Text([
        '   ',
        ('key', 'Q'), ' -- Quit    ',
        ('key', 'F1'), ' -- Help   ',
        ('key', 'Ctrl+R'), ' -- Reload directory    ',
        ('key', 'F5'), ' -- Run script   ',
        ('key', 'F6'), ' -- Kill %d subprocesses   ',
        ]), 'header2')
    ])

top = urwid.Frame(kolumny, head, footer)

#map2 = urwid.AttrMap(fill, 'streak')
main_loop = urwid.MainLoop(top, unhandled_input = ui, palette = palette)
main_loop.run()

