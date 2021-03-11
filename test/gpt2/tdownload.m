classdef(SharedTestFixtures = {DownloadGPT2Fixture}) tdownload < matlab.unittest.TestCase
    % tdownload   Tests for gpt2.download
    
    % Copyright 2020 The MathWorks, Inc.
    
    % downloadGPT2Fixture.setup calls gpt2.download so this test is just a
    % sanity check that the required files are downloaded.
    
    properties(Constant)
        ExpectedFiles = ["gpt2_355M_params.mat","gpt2_vocab.bpe","gpt2_encoder.txt"]
    end
    
    methods(Test)
        function verifyFilesExist(test)
            aFile = gpt2.internal.getSupportFilePath('gpt2_vocab.bpe');
            directory = fileparts(aFile);
            files = dir(directory);
            filenames = {files.name};
            % Check that the expected files were downloaded by the fixture.
            % Do not test these are the only files since we now use the
            % support files location, and what files exist there depend on
            % the user.
            import matlab.unittest.constraints.IsSupersetOf
            test.verifyThat(filenames,IsSupersetOf(test.ExpectedFiles),...
                "Expected files not downloaded for gpt2-355M.");
        end
    end
end
        