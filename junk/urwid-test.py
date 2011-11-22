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
        kolumny.set_focus(2)
        return
    if input == 'ctrl r':
        raise urwid.ExitMainLoop()
    if input == 'enter':
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
        self._w = urwid.AttrWrap(self._w, None)
        self._w.attr = 'banner'
        self._w.focus_attr = 'header2'
    def get_display_text(self):
        return self.get_node().get_key()
    def selectable(self):
        return True
    def keypress(self, size, key):
        key = self.__super.keypress(size, key)
        if key == 'enter':
            if self.is_leaf:
                if os.path.isfile('../' + fm.get_focus()[1].get_key()):
                    s = open('../'+fm.get_focus()[1].get_key()).read()
                    output.set_edit_text(s)
                else:
                    os.system('ls -l ../' + fm.get_focus()[1].get_key())
            else:
                self.expanded = not self.expanded
                self.update_expanded_icon()
        elif key == 'j':
            head.original_widget.set_text(str(fm.get_focus()))
        return key


class FileNode(urwid.TreeNode):
    def __init__(self, path, parent = None):
        urwid.TreeNode.__init__(self, path, key = path, parent = parent)
    def load_widget(self):
        return TreeNodeWidget(self)

class DirectoryNode(urwid.ParentNode):
    def __init__(self, path, parent = None):
        urwid.ParentNode.__init__(self, path, key = '/', parent = parent, depth = 0)
    def load_child_keys(self):
        return sorted(filter(lambda f: os.path.isdir('../'+f), os.listdir('..'))) + \
                sorted(filter(lambda f: os.path.isfile('../'+f), os.listdir('..')))
    def load_child_node(self, key):
        return FileNode(key, parent = self)
    def load_widget(self):
        return TreeNodeWidget(self)

class FM(urwid.TreeListBox):
    def keypress(self, size, key):
        self.__super.keypress(size, key)
        return key

fm = FM(urwid.TreeWalker(DirectoryNode('/')))
output = urwid.Edit('')

kolumny = urwid.Columns([
    ('weight', 1, urwid.AttrMap(fm, 'fm')),
    ('fixed', 1, urwid.Filler(urwid.Text('|' * 500), 'top')),
    ('weight', 3, urwid.ListBox(urwid.SimpleListWalker([
        urwid.Text('Output:'),
        urwid.Divider('-'),
        output,
        urwid.Divider('-'),
        ])) ),
    ], focus_column=2, dividechars=1)

top = urwid.Frame(kolumny, head, urwid.AttrMap(urwid.Text([
    ('key', 'Q'), ' -- Quit    ', ('key', 'F1'), ' -- help   ', ('key', 'R'), ' -- reload directory']), 'header'))

#map2 = urwid.AttrMap(fill, 'streak')
loop = urwid.MainLoop(top, unhandled_input = ui, palette = palette)
loop.run()

