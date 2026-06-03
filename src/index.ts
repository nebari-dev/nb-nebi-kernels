import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { Notification } from '@jupyterlab/apputils';

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';

import { IMissingInfo, missingKernelInfo } from './missingKernel';

/**
 * Raise a non-auto-closing warning describing the missing kernel, with a
 * "copy install command" action when the server supplied one.
 */
function notifyMissing(info: IMissingInfo): void {
  const parts = [
    `The pixi environment "${info.env}" (workspace "${info.workspace}") has ` +
      'no Jupyter kernel installed, so this notebook cannot run code yet.'
  ];
  if (info.command) {
    parts.push(`Install one with:\n\n${info.command}`);
  } else {
    parts.push(
      'Install a kernel package (e.g. ipykernel) into the env, then reselect ' +
        'the kernel.'
    );
  }

  const actions: Notification.IAction[] = [];
  if (info.command) {
    const command = info.command;
    actions.push({
      label: 'Copy install command',
      callback: event => {
        // Keep the notification open so the user can act on it.
        event.preventDefault();
        void navigator.clipboard?.writeText(command);
      }
    });
  }

  Notification.warning(parts.join('\n\n'), { autoClose: false, actions });
}

/**
 * Frontend plugin: watch notebook kernel selection and proactively warn when
 * the chosen kernel is a Nebi stub for a pixi env with no Jupyter kernel,
 * instead of making the user discover it by running a cell.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'nb-nebi-kernels:missing-kernel-notifier',
  description:
    'Warns when a notebook selects a Nebi kernel whose pixi env has no ' +
    'Jupyter kernel installed.',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker): void => {
    const specsManager = app.serviceManager.kernelspecs;
    // Avoid re-warning for a kernel we've already flagged on a given panel.
    const lastNotified = new WeakMap<NotebookPanel, string>();

    const check = async (panel: NotebookPanel): Promise<void> => {
      await specsManager.ready;
      if (panel.isDisposed) {
        return;
      }
      const session = panel.sessionContext;
      const kernelName =
        session.session?.kernel?.name ?? session.kernelPreference.name;
      const info = missingKernelInfo(specsManager.specs, kernelName);
      if (!info || !kernelName) {
        return;
      }
      if (lastNotified.get(panel) === kernelName) {
        return;
      }
      lastNotified.set(panel, kernelName);
      notifyMissing(info);
    };

    tracker.widgetAdded.connect((_, panel) => {
      // Re-check on every kernel change (covers the kernel picker dialog) and
      // once when the session first becomes ready (covers opening a notebook
      // already bound to a stub kernel).
      panel.sessionContext.kernelChanged.connect(() => void check(panel));
      void panel.sessionContext.ready
        .then(() => check(panel))
        .catch(() => {
          /* session never became ready; nothing to warn about */
        });
    });
  }
};

export default plugin;
