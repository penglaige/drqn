# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

from __future__ import print_function

# Allow MalmoPython to be imported both from an installed
# malmo module and (as an override) separately as a native library.
try:
    import MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython

import os
import sys
import errno

def fix_print():
    # We want to flush the print output immediately, so that we can view test output as it happens.
    # The way to do this changed completely between Python 2 and 3, with the result that setting this
    # in a cross-compatible way requires a few lines of ugly code.
    # Rather than include this mess in every single sample, it's nice to wrap it into a handy
    # function - hence this.
    if sys.version_info[0] == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
    else:
        import functools
        # Have to assign to builtins or the change won't make it outside of this module's scope
        import builtins
        builtins.print = functools.partial(print, flush=True)

def parse_command_line(agent_host, argv=None):
    if argv is None:
       argv = sys.argv
    # Add standard options required by test suite:
    agent_host.addOptionalStringArgument( "recording_dir,r", "Path to location for saving mission recordings", "./records" )
    agent_host.addOptionalStringArgument( "test_file", "Path of the model to test", "" )
    agent_host.addOptionalFlag( "test_model", "1:test the model")
    agent_host.addOptionalFlag( "train", "1:train 0:small_test")
    agent_host.addOptionalFlag( "record_video,v", "Record video stream" )
    agent_host.addOptionalFlag( "gpu", "Whether use a gpu" )
    agent_host.addOptionalFlag( "dqn", "Whether use a dqn" )
    agent_host.addOptionalFlag( "random", "Whether use a random player" )
    agent_host.addOptionalFlag( "double_dqn", "Whether use a double dqn")
    agent_host.addOptionalFlag( "dueling_dqn"," Whether use a dueling dqn")

    # Attempt to parse:
    try:
        agent_host.parse(argv)
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)


def get_video_xml(agent_host):
    return '<VideoProducer><Width>860</Width><Height>480</Height></VideoProducer>' if agent_host.receivedArgument("record_video") else ''

def get_default_recording_object(agent_host, filename):
    # Convenience method for setting up a recording object - assuming the recording_dir and record_video
    # flags were passed in as command line arguments (see parse_command_line above).
    # (If no recording destination was passed in, we assume no recording is required.)
    my_mission_record = MalmoPython.MissionRecordSpec()
    recordingsDirectory = get_recordings_directory(agent_host)
    if recordingsDirectory:
        my_mission_record.setDestination(recordingsDirectory + "//" + filename + ".tgz")
        my_mission_record.recordRewards()
        my_mission_record.recordObservations()
        my_mission_record.recordCommands()
        if agent_host.receivedArgument("record_video"):
            my_mission_record.recordMP4(24,2000000)
    return my_mission_record

def get_recordings_directory(agent_host):
    # Check the dir passed in:
    recordingsDirectory = agent_host.getStringArgument('recording_dir')
    if recordingsDirectory:
        # If we're running as an integration test, we want to send all our recordings
        # to the central test location specified in the environment variable MALMO_TEST_RECORDINGS_PATH:
        if agent_host.receivedArgument("test"):
            try:
                test_path = os.environ['MALMO_TEST_RECORDINGS_PATH']
                if test_path:
                    recordingsDirectory = os.path.join(test_path, recordingsDirectory)
            except:
                pass
        # Now attempt to create the folder we want to write to:
        try:
            os.makedirs(recordingsDirectory)
        except OSError as exception:
            if exception.errno != errno.EEXIST: # ignore error if already existed
                raise
    return recordingsDirectory


def get_options(agent_host):
    # Get other optional arguments
    gpu = agent_host.receivedArgument("gpu")
    double_dqn = agent_host.receivedArgument("double_dqn")
    dueling_dqn = agent_host.receivedArgument("dueling_dqn")
    train = agent_host.receivedArgument("train")
    dqn = agent_host.receivedArgument("dqn")
    random_play = agent_host.receivedArgument("random")
    test_model = agent_host.receivedArgument("test_model")
    modelFile = agent_host.getStringArgument('test_file')
    return train, test_model, gpu, dqn, double_dqn, dueling_dqn, random_play, modelFile
