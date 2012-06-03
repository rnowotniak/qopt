#!/usr/bin/python


if __name__ == "__main__":

    print "\n[Nicessa] Warning: There have been some changes in Nicessas structure due to ticket #6 (The experiment metaphor is outdated)\n"\
          "You are seeing this warning because you called the (deprecated) file experiment.py. Call nicessa.py instead.\n"\
          "\n"\
          "A list of changes:\n"\
          "  * experiment.py (the main script) is now called nicessa.py\n"\
          "  * Nicessa now expects to find a configuration file called nicessa.conf instead of experiment.conf\n"\
          "  * The 'vars'-setting is now called 'parameters'\n"\
          "  * The 'experiments'-setting is now called 'simulations'"

