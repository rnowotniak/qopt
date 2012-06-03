#!/usr/bin/python

'''
plotter
=======
'''

import sys
import os
import os.path as osp
from subprocess import Popen
from shutil import rmtree

import compressor
import harvester


with_names = False # if to have title in PDF
show_pdfs = False  # if to show PDF right after creation
pdf_viewer = 'okular'
tmp_dir = 'tmp_plotter'


def plot(filepath='', delim=',', outfile_name='', name='My simulation',\
         xcol=1, x_label='iteration', y_label='value', x_range=None,  y_range='[0:10]',\
         use_y_errorbars=False, errorbar_every=1, infobox_pos='top left', use_colors=True,\
         use_tex=False, line_width=6, font_size=22, custom_script='', plots=[]):
    '''
    Make plots for specific portions of data in one figure.

    For each graph, this function ...

    - selects folders with data files depending on parameter values you provide
    - collects all log files contained in them in a temporary folder
    - averages over the contents or just selects values from them
    - makes a (gnu)plot out of that, with yerrorbars if you want

    The output is one PDF file.
    Creating PDF for this yields far nicer linetypes.
    Also, this makes different linetypes (e.g. dashes/dots) possible,
    bcs papers are often printed b/w.

    In addition to gnuplot, you need epstopdf installed.

    :param string filepath: path to data
    :param string delim: delimiter used between columns
    :param string outfile_name: name of the PDF file you want to make
    :param string name: Title of the simulation
    :param int xcol: column >= 1
    :param string x_label: label on x axis
    :param string y_label: label on y axis
    :param string x_range: the range of values for x axis, in the form of
           "[a:b]", defaults to None, meaning that gnuplot should take to the actual
           value range of x values
    :param string y_range: the range of values for y axis, defaults to '[0:10]'
    :param boolean use_y_errorbars: True if errorbars should be shown, default is False
    :param int errorbar_every: show an errorbar every x steps, defaults to 1
    :param string infobox_pos: where the infobox should go, e.g. 'bottom left',
           defaults to 'top left' when given empty
    :param boolean use_colors: whether to use colors or be b+w, defaults True
    :param boolean use_tex: whether to use enhanced mode, which interpretes
           tex-like encoding (e.g. subscript, math, greek symbols), defaults to False
    :param int line_width: line width in pixels, defaults to 6
    :param int font_size: font size (a number), defaults to 22
    :param list plots: list of plot descriptions, defaults to empty list
    '''

    print '[Nicessa] Preparing %s: ' % outfile_name ,

    # make sure tmp dir exists
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # ---- collect relevant data for each requested plot ----
    # get relevant files
    searches = {}
    for p in plots:
        # make sure no old data is around
        plot_dir = '%s/%s' % (tmp_dir, p['_name'])
        if os.path.exists(plot_dir):
            if len(os.listdir(plot_dir)) > 0:
                Popen('rm -r %s/*' % plot_dir, shell=True).wait()
        searches[p['_name']] = [(k, p[k]) for k in p.keys() if not k.startswith('_')]
        print '%s ' % p['_name'],
    failed = harvester.collect_files(searches, filepath, tmp_dir)
    print
    # handle errors
    init_plots_num = len(plots)
    if len(failed) > 0:
        print "[Nicessa] WARNING: Selectors %s didn't match any folders!" % ','.join(failed)
        for fail in failed:
            for p in plots:
                if p['_name'] == fail:
                    plots.remove(p)
    if len(failed) == init_plots_num:
        print "[Nicessa] In fact, no selectors of this figure matched anything. Aborting ..."
        print
        return

    # ---- prepare data  ----
    for p in plots:
        if p['_type'] == 'line':
            compressor.avg_stats(xcol, int(p['_ycol']), None, filePrefix='log',
                      fileSuffix='.dat', filePath='%s/%s' % (tmp_dir, p['_name']),
                      delim=delim, outName='%s/%s/all.dat' % (tmp_dir, p['_name']))
        else: # scatter
            if not p.has_key('_select'):
                p['_select'] = 'all'
            harvester.collect_values('%s/%s' % (tmp_dir, p['_name']), delim, '%s/%s/all.dat' \
                           % (tmp_dir, p['_name']), cols=[xcol, int(p['_ycol'])],
                           selector=p['_select'] )

    # ---- plot the results ----
    if custom_script != "":
        if not osp.exists(custom_script):
            print "[Nicessa] Cannot find custom script at [%s]. Aborting ..." % (custom_script)
            print
            return
        print '[Nicessa] Using custom script at %s' % custom_script
        Popen('cp %s %s/plot.gnu' % (custom_script, tmp_dir), shell=True).wait()
    else:
        e = ''
        if use_tex:
            e = 'enhanced'

        c = 'monochrome'
        if use_colors:
            c = 'color'

        # write gnuplot code
        gnu = "set terminal postscript %s eps %s dashed lw %d rounded %d;\n" % (e, c, int(line_width), int(font_size))
        gnu += "set output '%s.eps';\n" % name
        if x_range:
            gnu += "set xrange %s;\n" % x_range
        gnu += "set yrange %s;\n" % y_range
        if with_names:
            gnu += "set title '%s';" % name
        gnu += "set xlabel '%s';\n" % x_label
        gnu += "set ylabel '%s';\n" % y_label
        gnu += "set key %s spacing 1;\n" % infobox_pos
        gnu += "plot "
        # first the actual plots
        num = 1
        for p in plots:
            smu = ''
            if p['_type'] == 'line':
                smu = 'smooth unique'
            gnu += "'%s/all.dat' %s " % (p['_name'], smu)
            gnu += " title '%s' lt %d" % (p['_name'], num)
            if num < len(plots):
                gnu += ','
            num += 1
        # then the errorbars (this way, they don't reserve the linestyles)
        if use_y_errorbars:
            offset = 1
            offset_step = max(1, int(errorbar_every) / 5)
            offtxt = ''
            if int(errorbar_every) > 1:
                offtxt = 'every %d::%d' % (int(errorbar_every), offset)
            num = 1
            gnu += ","
            for p in plots:
                gnu += "'%s/all.dat' %s with yerrorbars title '' lt %d" \
                        % (p['_name'], offtxt, num)
                if num < len(plots):
                    gnu += ','
                num += 1
                #TODO: this isn't nice for everyone... can maybe be derived from xrange somehow ... ?
                offset += offset_step

        # execute gunplot code
        gnuf = open('%s/plot.gnu' % tmp_dir, 'w')
        gnuf.write(gnu)
        gnuf.close()

    # generate PDF and maybe show it
    print '[Nicessa] Plotting %s' % outfile_name
    Popen('cd %s; gnuplot plot.gnu; epstopdf %s.eps; cd ..' % (tmp_dir, name), shell=True).wait()
    Popen('cp %s/%s.pdf %s' % (tmp_dir, name, outfile_name), shell=True).wait()
    print

    if osp.exists(tmp_dir) and not '-k' in sys.argv:
        rmtree(tmp_dir)

    if show_pdfs:
        Popen('%s %s' % (pdf_viewer, outfile_name), shell=True).wait()

