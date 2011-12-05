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
import re

import urwid

palette = [
        ('normal', 'dark gray', 'black'),
        ('banner', 'black', 'light gray', 'standout,underline'),
        ('header', 'yellow', 'dark blue', 'bold,standout,underline'),
        ('header2', 'yellow', 'dark red', 'bold,standout,underline'),
        ('streak', 'black', 'dark red', 'standout'),
        ('fm', 'white', 'black', 'standout'),
        ('fm_entry', 'light gray', 'black'),
        ('fm_dir', 'white', 'black'),
        ('fm_selected', 'black', 'light green'),
        ('footer', 'light green', 'dark blue', 'bold'),
        ('key', 'light green', 'dark blue', 'bold'),
        ('divider', 'light green', 'dark blue', 'bold'),
        ('main', 'white', 'black', 'bold'),
        ('bg', 'black', 'dark blue'),]

def ui(input):
    if input == 'ctrl e':
        kolumny.set_focus(0)
        return
    elif input == 'ctrl r':
        kolumny.widget_list[0] = urwid.AttrMap(FM(urwid.TreeWalker(DirectoryNode('.'))), 'fm')
        preview.set_edit_text('')
    elif input == 'f6':
        global subprocesses
        for p in subprocesses:
            main_loop.remove_watch_file(p.stdout.fileno())
            os.kill(p.pid, 9)
        subprocesses = []
        footer1.set_text(footer1_txt + str(subprocesses))
    elif input == 'f8':
        raise urwid.ExitMainLoop()
    elif input == 'ctrl c':
        raise urwid.ExitMainLoop()
    elif input == 'enter':
        txt.set_text(('header', 'value: ' + ask.edit_text))
        ask.edit_text = ''
        return
    elif type(input) in (type(()), type([])):
        # mouse press
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

head = urwid.AttrMap(urwid.Text('Quantum-Inspired Evolutionary Algorithms framework (C) Robert Nowotniak, 2011', align='center', wrap='clip'), 'header2')


class LineWalker(urwid.ListWalker):
    """ListWalker-compatible class for lazily reading file contents."""
    
    def __init__(self, name):
        self.file = open(name)
        self.lines = []
        self.focus = 0
    
    def get_focus(self): 
        return self._get_at_pos(self.focus)
    
    def set_focus(self, focus):
        self.focus = focus
        self._modified()
    
    def get_next(self, start_from):
        return self._get_at_pos(start_from + 1)
    
    def get_prev(self, start_from):
        return self._get_at_pos(start_from - 1)

    def read_next_line(self):
        """Read another line from the file."""
        
        next_line = self.file.readline()
        
        if not next_line or next_line[-1:] != '\n':
            # no newline on last line of file
            self.file = None
        else:
            # trim newline characters
            next_line = next_line[:-1]

        expanded = next_line.expandtabs()
        try:
            expanded = unicode(expanded)
        except Exception:
            expanded = re.sub(r'(?i)[^\000-\176]', '?', expanded)
        
        edit = urwid.Edit("", expanded, allow_tab=True, wrap = 'clip')
        edit.set_edit_pos(0)
        edit.original_text = next_line
        self.lines.append(edit)

        return next_line
        
    
    def _get_at_pos(self, pos):
        """Return a widget for the line number passed."""
        
        if pos < 0:
            # line 0 is the start of the file, no more above
            return None, None
            
        if len(self.lines) > pos:
            # we have that line so return it
            return self.lines[pos], pos

        if self.file is None:
            # file is closed, so there are no more lines
            return None, None

        assert pos == len(self.lines), "out of order request?"

        self.read_next_line()
        
        return self.lines[-1], pos
    
    def split_focus(self):
        """Divide the focus edit widget at the cursor location."""
        
        focus = self.lines[self.focus]
        pos = focus.edit_pos
        edit = urwid.Edit("",focus.edit_text[pos:], allow_tab=True)
        edit.original_text = ""
        focus.set_edit_text(focus.edit_text[:pos])
        edit.set_edit_pos(0)
        self.lines.insert(self.focus+1, edit)

    def combine_focus_with_prev(self):
        """Combine the focus edit widget with the one above."""

        above, ignore = self.get_prev(self.focus)
        if above is None:
            # already at the top
            return
        
        focus = self.lines[self.focus]
        above.set_edit_pos(len(above.edit_text))
        above.set_edit_text(above.edit_text + focus.edit_text)
        del self.lines[self.focus]
        self.focus -= 1

    def combine_focus_with_next(self):
        """Combine the focus edit widget with the one below."""

        below, ignore = self.get_next(self.focus)
        if below is None:
            # already at bottom
            return
        
        focus = self.lines[self.focus]
        focus.set_edit_text(focus.edit_text + below.edit_text)
        del self.lines[self.focus+1]


class TreeNodeWidget(urwid.TreeWidget):
    def __init__(self, node):
        self.__super.__init__(node)
        if hasattr(self._w.original_widget, 'set_wrap_mode'):
            self._w.original_widget.set_wrap_mode('clip')
        self._w = urwid.AttrWrap(self._w, None)
        if os.path.isdir(node.get_key()):
            self._w.attr = 'fm_dir'
            self.expanded = node.get_depth() == 0 or \
                    node.get_key().split('/')[-1] in \
                    ('CUDA', 'junk', 'benchmarks', 'PL-GRID', 'experiments', 'contrib')
            self.update_expanded_icon()
        else:
            self._w.attr = 'fm_entry'
        self._w.focus_attr = 'fm_selected'
    def get_display_text(self):
        if self.get_node().get_depth() == 0:
            n = 'QOPT'
        else:
            n = os.path.basename(self.get_node().get_key())
        if self.get_node().__class__ == DirectoryNode:
            return '[%s/]' % n
        else:
            return n
    def selectable(self):
        return True
    def keypress(self, size, key):
        key = self.__super.keypress(size, key)
        if key == 'r':
            preview.set_edit_text('')
            f = fm.get_focus()[1].get_key()
            p = subprocess.Popen([f], shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocesses.append(p)
            footer1.set_text(footer1_txt + str(subprocesses))
            def cb():
                line = p.stdout.readline()
                if line == '':
                    main_loop.remove_watch_file(handle)
                    return
                preview.set_edit_text(preview.get_edit_text() + line)
            handle = main_loop.watch_file(p.stdout.fileno(), cb)
        elif key == 'enter':
            preview.set_edit_text('')
            if self.is_leaf:
                f = fm.get_focus()[1].get_key()
                if f.endswith('.py'):
                    p = subprocess.Popen(["python", f], shell=False,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    subprocesses.append(p)
                    footer1.set_text(footer1_txt + str(subprocesses))
                    def cb():
                        line = p.stdout.readline()
                        if line == '':
                            main_loop.remove_watch_file(handle)
                            return
                        preview.set_edit_text(preview.get_edit_text() + line)
                    handle = main_loop.watch_file(p.stdout.fileno(), cb)
                else:
                    s = open(f).read()
                    preview.set_edit_text(s)
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
        c = filter(lambda f: f not in ('.git', 'SdkMasterLog.csv', 'deviceQuery.txt'), c)
        c = filter(lambda f: not f.startswith('.'), c)
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
        if key == 'v':
            f = self.get_focus()[1].get_key()
            if not os.path.isdir(f):
                walker = LineWalker(f)
                preview.set_edit_text(str(kolumny.widget_list[2].original_widget.body[2]))
                kolumny.widget_list[2].original_widget.body[2] = urwid.BoxAdapter(urwid.ListBox(walker), 10)
                #kolumny.widget_list[2].original_widget.body[2] = urwid.BoxAdapter(walker, 10)
                # s = open(f).read()
                # preview.set_edit_text(s)
        elif key == 'E':
            #os.spawnlp(os.P_NOWAIT, 'xterm', 'xterm', '-e', 'vim "%s"' % str(self.get_focus()[1].get_key()))
            os.system('bash -c "xterm&disown"')# % str(self.get_focus()[1].get_key()))
        self.__super.keypress(size, key)
        return key


subprocesses = []

fm = FM(urwid.TreeWalker(DirectoryNode('.')))
preview = urwid.Edit('', wrap='clip', multiline=True)
output = urwid.Edit('', wrap='clip', multiline=True)

kolumny = urwid.Columns([
    ('fixed', 30, urwid.AttrMap(fm, 'fm')),
    ('fixed', 1, urwid.AttrMap(urwid.Filler(urwid.Text(' ' * 5), 'top'), 'header2' )),
    ('weight', 3, urwid.AttrMap(urwid.ListBox(urwid.SimpleListWalker([
        urwid.Text('Preview:'),
        urwid.Divider('-'),
        urwid.BoxAdapter(urwid.Filler(preview, 'top'), 10),
        urwid.Divider('-'),
        urwid.Text('Output:'),
        urwid.Divider('-'),
        output,
        urwid.Divider('-'),
        ])), 'main')),
    ], focus_column=0, dividechars=0)


footer1_txt = 'Subprocesses: '
footer1 = urwid.Text('')
footer1.set_text(footer1_txt + str(subprocesses))

footer = urwid.Pile([
    urwid.AttrMap(footer1, 'header2'),
    urwid.AttrMap(urwid.Text([
        ('key', 'Q'),      ':Quit ',
        ('key', 'e'),      ':edit ',
        ('key', 'E'),      ':edit in new window ',
        ('key', 'F1'),     ':Help ',
        ('key', 'Ctrl+R'), ':Reload directory ',
        ('key', 'F5'),     ':Run script ',
        ('key', 'F6'),     ':Kill subprocesses ',
        ], wrap='clip'), 'footer')
    ])

top = urwid.Frame(kolumny, head, footer)

#map2 = urwid.AttrMap(fill, 'streak')
main_loop = urwid.MainLoop(top, unhandled_input = ui, palette = palette)
main_loop.run()

