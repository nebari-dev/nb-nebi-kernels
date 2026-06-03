import { missingKernelInfo } from '../missingKernel';

function specsWith(metadata: Record<string, unknown> | undefined): any {
  return {
    default: 'nebi-ws-env',
    kernelspecs: {
      'nebi-ws-env': {
        name: 'nebi-ws-env',
        display_name: 'ws (env)',
        language: 'no-op',
        argv: [],
        resources: {},
        metadata
      }
    }
  };
}

describe('missingKernelInfo', () => {
  it('returns null when kernelspecs are not loaded', () => {
    expect(missingKernelInfo(null, 'nebi-ws-env')).toBeNull();
  });

  it('returns null when no kernel name is given', () => {
    const specs = specsWith({ nebi_kernel_state: 'missing-kernel' });
    expect(missingKernelInfo(specs, undefined)).toBeNull();
  });

  it('returns null for a working (ready) kernel', () => {
    const specs = specsWith({ nebi_kernel_state: 'ready' });
    expect(missingKernelInfo(specs, 'nebi-ws-env')).toBeNull();
  });

  it('returns null when the kernelspec has no metadata', () => {
    expect(missingKernelInfo(specsWith(undefined), 'nebi-ws-env')).toBeNull();
  });

  it('extracts workspace, env and install command for a stub kernel', () => {
    const command =
      'pixi add --manifest-path /home/user/ds/pixi.toml --feature gpu ipykernel';
    const specs = specsWith({
      nebi_kernel_state: 'missing-kernel',
      nebi_workspace: 'data-science',
      pixi_environment: 'gpu',
      nebi_install_command: command
    });
    expect(missingKernelInfo(specs, 'nebi-ws-env')).toEqual({
      workspace: 'data-science',
      env: 'gpu',
      command
    });
  });

  it('falls back gracefully when optional fields are missing', () => {
    const specs = specsWith({ nebi_kernel_state: 'missing-kernel' });
    expect(missingKernelInfo(specs, 'nebi-ws-env')).toEqual({
      workspace: 'this workspace',
      env: 'this environment',
      command: null
    });
  });
});
