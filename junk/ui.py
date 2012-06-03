#!/usr/bin/python
#
# QOPT GUI in URWID
# Copyright (C) 2011   Robert Nowotniak <robert@nowotniak.com>
#

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


import sys
import os
import subprocess
import re
import fnmatch
import urwid


# LineWalker {{{
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
# }}}

# File Manager classes {{{
class FM(urwid.TreeListBox):
    IGNORE_PATTERNS = []

    def __init__(self, value):
        urwid.TreeListBox.__init__(self, value)

        f = open('config/ui-ignore')
        FM.IGNORE_PATTERNS = [p.strip() for p in f.readlines()]
        f.close()

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
            ui.main_loop.draw_screen()
        if key == 'v':
            f = self.get_focus()[1].get_key()
            if not os.path.isdir(f):
                walker = LineWalker(f)
                ui.main_columns.widget_list[2].original_widget.body[2] = urwid.BoxAdapter(urwid.ListBox(walker), 5)
        elif key == 'E':
            os.system('bash -c "xterm&disown"')# % str(self.get_focus()[1].get_key()))
        elif key in [str(c) for c in xrange(1,10)]:
            ui.nparallel = int(key)
            ui.debug_footer.set_text('Number of parallel processes: ' + str(ui.nparallel))
            for n in xrange(len(ui.progress_bars)):
                ui.slw.pop(5) # XXX make it more flexible
            ui.progress_bars = []
            for n in xrange(ui.nparallel):
                pb = urwid.ProgressBar('pg normal', 'pg complete', 0, 1)
                ui.progress_bars.append(pb)
                ui.slw.insert(5, pb)
            # insert ResultButtons GridFlow
            ui.resultsRadioButtons = []
            ui.slw.pop(6 + ui.nparallel) # XXX make it more flexible
            ui.slw.insert(6 + ui.nparallel, urwid.GridFlow([
                urwid.AttrMap(
                    ResultButton(ui.resultsRadioButtons, str(n), ui.on_result_button), \
                            'button normal', 'button select') \
                            for n in xrange(1, ui.nparallel + 1)], 5, 2, 0, 'left'))
        self.__super.keypress(size, key)
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
        c = map(lambda f: os.path.join(self.get_key(), f), c)  # full paths
        c = map(lambda f: f.replace('./', '', 1), c) # paths relative to main qopt dir
        for pat in FM.IGNORE_PATTERNS:
            c = filter(lambda f: not re.match(pat, f), c)
        c = map(lambda f: os.path.join(os.path.dirname(__file__), f), c) # full paths
        return c
    def load_child_node(self, key):
        if os.path.isdir(key):
            return DirectoryNode(key, parent = self, depth = self.get_depth() + 1)
        return FileNode(key, parent = self, depth = self.get_depth() + 1)
    def load_widget(self):
        return TreeNodeWidget(self)

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
                    ('CUDA', 'benchmarks', 'PL-GRID', 'EXPERIMENTS', 'contrib', \
                    # 'junk', \
                    'algorithms')
            self.update_expanded_icon()
        elif os.access(node.get_key(), os.X_OK):
            self._w.attr = 'fm_executable'
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
        elif n.endswith('.py'):
            return n.rsplit('.', 1)[0] + ' [P]'
        else:
            return n
    def selectable(self):
        return True
    def keypress(self, size, key):
        key = self.__super.keypress(size, key)
        if key == 'enter':
            if len(ui.progress_bars) == 0:
                ui.debug_footer.set_text('Set the number of processes first!')
                return
            f = ui.fm.get_focus()[1].get_key()
            ui.basename = os.path.basename(f)  # XXX fix this
            if os.path.isdir(f):
                self.expanded = not self.expanded
                self.update_expanded_icon()
            elif os.access(f, os.X_OK): # execute this file  TODO: or .endswith('.py')
                ui.debug_footer.set_text('Executing %s in parallel %d times'% (f, ui.nparallel))
                for n in xrange(ui.nparallel):
                    proc = subprocess.Popen(['%s' % f], shell = False, close_fds = True, \
                            stdout = open(getResultsFilename(n + 1), 'w'), stderr=subprocess.STDOUT)
                            #stdout = open("%s/output.%d"%(ui.resultsDir,n+1), 'w'), stderr=subprocess.STDOUT)
                    ui.subprocesses.append(proc)
                    ui.progress_bars[n].set_completion(0)
                def cb(loop, data):
                    for p in ui.subprocesses:
                        # check if the subprocess has finished already
                        if type(p.poll()) == int:
                            ui.subprocesses.remove(p)
                    for n in xrange(0, ui.nparallel):
                        lines = tail(getResultsFilename(n+1), 20)
                        if n == map(lambda b: b.state, ui.resultsRadioButtons).index(True):
                            # ResultButton n is selected, so update output view accordingly
                            ui.output.set_edit_text(lines)
                        try:
                            # try to upgrade progress bar
                            evals = int( lines.split('\n')[-1].split()[0] )
                            proc = 1. * evals / ui.maxEvals
                            if proc > ui.progress_bars[n].current:
                                ui.progress_bars[n].set_completion(proc)
                        except Exception:
                            pass
                    ui.debug_footer.set_text(ui.debug_footer.get_text()[0] + ".")
                    if ui.subprocesses:
                        ui.main_loop.set_alarm_in(0.5, cb)
                    else:
                        ui.debug_footer.set_text('All the subprocesses have terminated.')
                ui.main_loop.set_alarm_in(0.5, cb)
        return key
# }}}


class ResultButton(urwid.RadioButton):
    def __init__(self, group, label, callback):
        urwid.RadioButton.__init__(self, group, label, on_state_change = callback)

    def keypress(self, size, key):
        if key == 'e':
            os.system('vim "%s"' % getResultsFilename(int(self.get_label())))
            ui.main_loop.draw_screen()
        return urwid.RadioButton.keypress(self, size, key)

# Main GUI class
class QOPTGui:
    palette = [
            ('normal', 'light gray', 'black'),
            ('banner', 'black', 'light gray', 'standout,underline'),
            ('header', 'yellow', 'dark blue', 'bold,standout,underline'),
            ('header2', 'yellow', 'dark red', 'bold,standout,underline'),
            ('streak', 'black', 'dark red', 'standout'),
            ('fm', 'white', 'black', 'standout'),
            ('fm_entry', 'light gray', 'black'),
            ('fm_dir', 'white', 'black'),
            ('fm_selected', 'black', 'light green'),
            ('fm_executable', 'light red', 'black'),
            ('footer', 'light green', 'dark blue', 'bold'),
            ('key', 'light green', 'dark blue', 'bold'),
            ('divider', 'light green', 'dark blue', 'bold'),
            ('bold', 'white', 'black', 'bold'),
            ('bg', 'black', 'dark blue'),
            ('button normal', 'light gray', 'dark blue'),
            ('button select', 'white', 'dark green'),
            ('pg normal', 'white', 'black', 'standout'),
            ('pg complete', 'white', 'dark green'),
            ('message', 'yellow', 'dark red'),
            ]

    def __init__(self):
        self.subprocesses = []
        self.nparallel = 1
        self.maxEvals = 100000
        self.resultsDir = '/tmp/qopt'

        self.fm = FM(urwid.TreeWalker(DirectoryNode('.')))
        self.head = urwid.AttrMap(urwid.Text('Quantum-Inspired Evolutionary Algorithms (C) Robert Nowotniak, 2011', align='center', wrap='clip'), 'header2')
        self.preview = urwid.Edit('', wrap='clip', multiline=True)
        self.output = urwid.Edit('', wrap='clip', multiline=True)
        self.debug_footer = urwid.Text('')
        self.footer1_txt = 'Subprocesses: '
        self.debug_footer.set_text(self.footer1_txt + str(self.subprocesses))
        self.footer = urwid.Pile([
            urwid.AttrMap(self.debug_footer, 'header2'),
            urwid.AttrMap(urwid.Text([
                ('key', 'Q'),      ':Quit ',
                ('key', 'enter'),  ':Execute ',
                ('key', 'e'),      ':edit ',
                ('key', 'E'),      ':edit in new window ',
                # ('key', 'F1'),     ':Help ',
                ('key', 'Ctrl+R'), ':Reload directory ',
                ('key', 'F6'),     ':Kill subprocesses ',
                ], wrap='clip'), 'footer')
            ])
        self.progress_bars = [ ]
        self.resultsRadioButtons = []
        self.slw = urwid.SimpleListWalker([
                urwid.AttrMap(urwid.Text('File preview:'), 'bold'),
                urwid.Divider('-'),
                urwid.BoxAdapter(urwid.Filler(self.preview, 'top'), 5),
                urwid.Divider('-'),
                urwid.AttrMap(urwid.Text('Progress:'), 'bold'),
                # ProgressBars are dynamically inserted here
                urwid.Divider('-'),
                urwid.GridFlow([], 5, 2, 0, 'left'), # GridFlow with ResultButtons is dynamically replaced
                urwid.Divider('-'),
                self.output,
                urwid.Divider('-'),
                ])
        self.main_columns = urwid.Columns([
            ('fixed', 30, urwid.AttrMap(self.fm, 'fm')),
            ('fixed', 1, urwid.AttrMap(urwid.Filler(urwid.Text(' ' * 5), 'top'), 'header2' )),
            ('weight', 3, urwid.AttrMap(urwid.ListBox(self.slw), 'normal')),
            ], focus_column=0, dividechars=0)
        self.topframe = urwid.Frame(self.main_columns, self.head, self.footer)

    def main(self):
        self.main_loop = urwid.MainLoop(self.topframe, unhandled_input = self.unhandled_input, palette = QOPTGui.palette)
        self.main_loop.run()

    def on_result_button(self, button, checked):
        if not checked:
            return
        try:
            data = tail(getResultsFilename(int(button.get_label())), 20)
            ui.output.set_edit_text(data)
        except Exception, e:
            ShowMessage(str(e), 'Error')

    def unhandled_input(self, input):
        if input == 'ctrl e':
            self.main_columns.set_focus(0)
            return
        if input == 'ctrl w':
            self.main_columns.set_focus(2)
            self.main_columns.widget_list[2].original_widget.set_focus(8 + len(self.progress_bars))
            return
        elif input == 'ctrl r':
            self.fm = FM(urwid.TreeWalker(DirectoryNode('.')))
            self.main_columns.widget_list[0] = urwid.AttrMap(self.fm, 'fm')
            self.preview.set_edit_text('')
        elif input == 'f6':
            for p in ui.subprocesses:
                os.kill(p.pid, 9)
                os.waitpid(p.pid, 0)
            self.subprocesses = []
            self.debug_footer.set_text('All subprocesses has been killed.')
        elif input == 'f1':
            self.main_loop.widget = \
                    urwid.Overlay(
                            urwid.LineBox( HelpDialog(), 'Help'),
                            self.topframe, 'center', 70, 'middle', 25)
        elif input == 'ctrl p':
            self.main_loop.widget = \
                    urwid.Overlay(
                            urwid.LineBox( ConfigDialog(), 'Configuration'),
                            self.topframe, 'center', 70, 'middle', 25)
        elif input == 'f2':
            ShowMessage('Blabla blablabla bla', 'Message foo')
        elif input == 'f8':
            raise urwid.ExitMainLoop()
        elif input == 'ctrl c':
            raise urwid.ExitMainLoop()
        elif type(input) in (type(()), type([])):
            # mouse press
            return

class ShowMessage():
    def __init__(self, msg, title = 'Message'):
        ui.main_loop.widget = \
                urwid.Overlay(urwid.AttrMap(urwid.LineBox(
                    urwid.ListBox(urwid.SimpleListWalker([
                        urwid.Text( msg ),
                        urwid.GridFlow([
                            urwid.AttrMap(urwid.Button('OK', self.on_ok_click),
                                'button normal', 'button select')], 10, 0, 0, 'center')
                        ])), title), 'message'),
                ui.topframe, 'center', 70, 'middle', msg.count('\n') + 6)
    def on_ok_click(self, d):
        ui.main_loop.widget = ui.topframe


class ConfigDialog(urwid.ListBox):
    def __init__(self):
        self.evalsIntEdit = urwid.IntEdit(default = ui.maxEvals)
        self.resultsDirEdit = urwid.Edit('', ui.resultsDir)
        urwid.ListBox.__init__(self,urwid.SimpleListWalker([
            urwid.Text('This is the configuration screen for QOpt.\n\n\n'),
            urwid.AttrMap(urwid.Text('Maximum number of evaluations:'), 'bold'),
            self.evalsIntEdit,
            urwid.Divider(' '),
            urwid.AttrMap(urwid.Text('Results directory:'), 'bold'),
            self.resultsDirEdit,
            urwid.GridFlow([
                urwid.AttrMap(urwid.Button('Close', self.on_close_click),
                    'button normal', 'button select')], 10, 0, 0, 'center')
            ]))
    def on_close_click(self, d):
        ui.main_loop.widget = ui.topframe
        try: ui.maxEvals = int(self.evalsIntEdit.get_edit_text())
        except Exception: pass
        try:
            os.mkdir(self.resultsDirEdit.get_edit_text())
            ui.resultsDir = self.resultsDirEdit.get_edit_text()
        except OSError, e:
            if e.errno != 17:
                raise
        ui.debug_footer.set_text('Max evals: %d' % ui.maxEvals)

class HelpDialog(urwid.ListBox):
    def __init__(self):
        urwid.ListBox.__init__(self,urwid.SimpleListWalker([
            urwid.Divider(' '),
            urwid.AttrMap(urwid.Text('Quantum Inspired Evolutionary Algorithms', 'center'), 'bold'),
            urwid.Text('Copyright (C) 2011 Robert Nowotniak <robert@nowotniak.com>', 'center'),
            urwid.Divider(' '),
            urwid.Divider('*'),
            urwid.Divider(' '),
            urwid.Text('   Keyboard shorcuts:\n'),
            urwid.Text(
                '      j,k    --  up/down in file editor\n'
                '        e    --  edit the file\n'
                '        E    --  edit the file in new window\n'
                '    enter    --  execute the file or do a special action\n'
                '      1-9    --  number of parallel processes\n'
                '       F1    --  this help screen\n'
                '       F6    --  kill all subprocesses\n'
                '   ctrl-e    --  back to file manager (left column)\n'
                '   ctrl-p    --  configuration dialog\n'
                '       F8    --  quit the application\n'
                ),
            urwid.GridFlow([
                urwid.AttrMap(urwid.Button('Close', self.on_close_click),
                    'button normal', 'button select')], 10, 0, 0, 'center')
            ]))
    def on_close_click(self, d):
        ui.main_loop.widget = ui.topframe

def getResultsFilename(n):
    try:
        subdir = "%s/%s"%(ui.resultsDir, ui.basename)
        os.mkdir(subdir)
    except Exception:
        pass
    return "%s/output.%d"%(subdir, n)

def tail(filename, nlines = 10):
    try:
        f = open(filename, 'r')
        f.seek(-1, 2)
        while True:
            ch = f.read(1)
            if ch != '\n':
                f.seek(-2, 1)
            else:
                break
        f.seek(-2, 1)
        endpos = f.tell() + 1
        n = 0
        while n < nlines:
            if f.read(1) == '\n':
                n += 1
            try:
                f.seek(-2, 1)
            except Exception:
                # no more lines before
                f.seek(-1, 1)
                break
        if n == nlines:
            f.seek(2, 1)
        result = f.read(endpos - f.tell())
        f.close()
    except Exception, e:
        #if e.errno == 2:
        return '(waiting for the results...)'
    return result


if __name__ == '__main__':
    ui = QOPTGui()
    ui.main()



# vim: set foldmethod=marker:
