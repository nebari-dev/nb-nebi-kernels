import type { KernelSpec } from '@jupyterlab/services';

/**
 * Value of the kernelspec metadata flag that `NebiKernelSpecManager` sets on a
 * stub kernel (a pixi env with no Jupyter kernel installed). Must stay in sync
 * with `nb_nebi_kernels.manager._stub_kernel_spec`.
 */
export const MISSING_STATE = 'missing-kernel';

export interface IMissingInfo {
  workspace: string;
  env: string;
  /** Copy-pasteable install command, when the server provided one. */
  command: string | null;
}

/**
 * If `kernelName` resolves to a Nebi stub kernelspec (one whose metadata
 * carries `nebi_kernel_state === "missing-kernel"`), return the details needed
 * to tell the user how to fix it. Otherwise return `null`.
 *
 * Kept free of `@jupyterlab/application` runtime imports so it can be unit
 * tested in isolation.
 */
export function missingKernelInfo(
  specs: KernelSpec.ISpecModels | null,
  kernelName: string | undefined | null
): IMissingInfo | null {
  if (!specs || !kernelName) {
    return null;
  }
  const spec = specs.kernelspecs[kernelName];
  const metadata = spec?.metadata as Record<string, unknown> | undefined;
  if (!metadata || metadata.nebi_kernel_state !== MISSING_STATE) {
    return null;
  }
  return {
    workspace:
      typeof metadata.nebi_workspace === 'string'
        ? metadata.nebi_workspace
        : 'this workspace',
    env:
      typeof metadata.pixi_environment === 'string'
        ? metadata.pixi_environment
        : 'this environment',
    command:
      typeof metadata.nebi_install_command === 'string'
        ? metadata.nebi_install_command
        : null
  };
}
