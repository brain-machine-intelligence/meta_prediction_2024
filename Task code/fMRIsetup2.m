
%% Preparation for serial communication for Lumina
IOPort('CloseAll');

ComInfo = [];

ComInfo.portSpec        = 'COM5'; % when we know which port will be used
ComInfo.joker           = '';
ComInfo.sampleFreq      = 120;
ComInfo.baudRate        = 115200;
ComInfo.specialSettings = [];
ComInfo.InputBufferSize = ComInfo.sampleFreq * 3600;
ComInfo.readTimeout     = min(max(10 * 1/ComInfo.sampleFreq, 15), 21);
ComInfo.settings        = sprintf('%s %s BaudRate=%i InputBufferSize=%i Terminator=0 ReceiveTimeout=%f ReceiveLatency=0.0001', ...
    ComInfo.joker, ComInfo.specialSettings, ComInfo.baudRate, ComInfo.InputBufferSize, ComInfo.readTimeout);

%% COM port open
try
    [myport, errmsg] = IOPort('OpenSerialPort', ComInfo.portSpec, ComInfo.settings);
catch
    error('Error during OpenSerialPort. Check the COM setting (e.g., port number).');
end

IOPort('ConfigureSerialPort', myport, sprintf('%s BlockingBackgroundRead=1 StartBackgroundRead=1', ComInfo.joker));

%% Wait for trigger
treceived = 0;
disp(treceived)
while treceived >=0
%     disp(treceived)
    try
        tic
        [pktdata, treceived] = IOPort('Read', myport, 1, 1); %pktdata를 kbcheck의 keycode처럼 쓰면 됩니다
        pktdata
        treceived
        toc
    catch
        error('Error detected while reading IOPort');
    end
end
