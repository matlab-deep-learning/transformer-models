classdef(SharedTestFixtures = {DownloadGPT2Fixture}) tdownload < matlab.unittest.TestCase
    % tdownload   Tests for gpt2.download
    
    % Copyright 2020 The MathWorks, Inc.
    
    % downloadGPT2Fixture.setup calls gpt2.download so this test is just a
    % sanity check that the required files are downloaded.
    
    properties(Constant)
        ExpectedDataDir = fullfile(getRepoRoot(),'gpt2-355M')
        ExpectedFiles = ["parameters.mat","vocab.bpe","encoder.txt"]
    end
    
    methods(Test)
        function verifyFilesExist(test)
            test.assertEqual(exist(test.ExpectedDataDir,"dir"),7,...
                "Expected download directory for gpt2-355M not created.");
            files = dir(test.ExpectedDataDir);
            filenames = {files.name};
            import matlab.unittest.constraints.IsSupersetOf
            test.verifyThat(filenames,IsSupersetOf(test.ExpectedFiles),...
                "Expected files not downloaded for gpt2-355M.");
            import matlab.unittest.constraints.IsSameSetAs
            % dir picks up "." and ".." too.
            test.verifyThat(setdiff(filenames,test.ExpectedFiles), IsSameSetAs([".",".."]),...
                "Unexpected files downloaded for gpt2-355M.");
        end
    end
end
        